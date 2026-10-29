# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DFlash draft with scratch tokens and an output-end path selector (AngelSpec E4 / E7).

AngelSpec (Scratch-Tokens-Impl.md sections 2, 7, 10) trains a shift-one DFlash drafter of ``S`` slots
whose block layout is ``[anchor, MASK x (S - 1)]`` (slot ``j`` predicts the ``j + 1``-th token after the
anchor, every slot drafts a token, like DSpark's ``sample_from_anchor``). On top of the DFlash backbone:

* **scratch tokens** (E7, ``num_scratch_tokens = r > 0``): every slot owns ``r`` extra query tokens that
  share its position, are never read out, and only add computation. A group (slot + scratch tokens)
  attends to the context and to groups ``0..j`` (``GroupCausalFlashAttentionBackend``). The input of a
  scratch token is its slot's input plus a learned vector ``scratch_embed[i]``.
* **path selector** (E4 / E7, ``selector_topk = K``): instead of the argmax of each slot's logits,
  the ``K`` best tokens of every slot are re-ranked given the predecessor chosen before them (a small MLP
  over ``K x K`` pairs, :class:`DFlashSelectorHead`) and the draft is the greedy walk through the tables.
  Everything is fixed-shape tensor math, so it runs inside the draft CUDA graph.

The draft proposes a deterministic path, so it is meant for greedy verification (temperature 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model


class DFlashSelectorHead(nn.Module):
    """AngelSpec ``OutputSelector`` (scorer ``mlp``) in inference form.

    For slot ``j`` and the ``K`` candidates ``a`` of slot ``j - 1``, ``b`` of slot ``j``::

        S_j(a, b) = l_j(b) + Delta([s_j; z_{j-1,a}; z_{j,b}; l_{j-1}(a); l_j(b)])

    ``l_j(c) = x_j . W_out[c]`` is the LM-head logit of candidate ``c`` (``x_j`` the slot's final-normed
    hidden), ``z_{j,c} = LN(W_fuse [LN(W_cand E_in[c]); LN(W_ctx x_j)])`` its representation and
    ``s_j = W_pair [x_{j-1}; x_j]`` a pair state shared by all ``K^2`` pairs. ``Delta`` is a two-layer
    MLP whose first layer is evaluated split by input (each candidate once, then broadcast-added).
    """

    def __init__(
        self, hidden_size: int, topk: int = 16, width: int = 128, mlp_hidden: int = 256
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.width = int(width)
        self.cand_proj = nn.Linear(hidden_size, width, bias=False)
        self.ctx_proj = nn.Linear(hidden_size, width, bias=False)
        self.cand_norm = nn.LayerNorm(width)
        self.ctx_norm = nn.LayerNorm(width)
        self.fuse = nn.Linear(2 * width, width, bias=False)
        self.z_norm = nn.LayerNorm(width)
        self.pair_proj = nn.Linear(2 * hidden_size, width, bias=False)
        # columns in the order [s; z_a; z_b; l_a; l_b]
        self.mlp1 = nn.Linear(3 * width + 2, mlp_hidden)
        self.mlp2 = nn.Linear(mlp_hidden, 1, bias=False)

    def encode(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``z``: ``x`` ``[N, D]`` (slot hidden), ``emb`` ``[N, K, D]`` (input embeddings of its candidates)."""
        cand = self.cand_norm(self.cand_proj(emb.to(x.dtype)))
        ctx = self.ctx_norm(self.ctx_proj(x)).unsqueeze(-2).expand_as(cand)
        return self.z_norm(self.fuse(torch.cat([cand, ctx], dim=-1)))

    def tables(
        self, x: torch.Tensor, lm_head_weight: torch.Tensor, embed_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidates ``[R, S, K]`` of every slot and the score tables ``[R, S - 1, K_a, K_b]``,
        ``tables[r, j - 1, a, b] = S_j(a, b)`` (fp32)."""
        num_reqs, num_slots, hidden = x.shape
        k, w = self.topk, self.width
        flat = x.reshape(num_reqs * num_slots, hidden)
        logits = flat @ lm_head_weight.t()
        cand = logits.topk(k, dim=-1).indices  # sorted, best first
        ell = logits.gather(1, cand).float()
        z = self.encode(flat, F.embedding(cand, embed_weight))
        cand3 = cand.view(num_reqs, num_slots, k)
        ell3 = ell.view(num_reqs, num_slots, k)
        z3 = z.view(num_reqs, num_slots, k, w)

        s = self.pair_proj(torch.cat([x[:, :-1], x[:, 1:]], dim=-1))  # [R, S-1, w]
        weight, bias = self.mlp1.weight, self.mlp1.bias
        base = F.linear(s, weight[:, :w], bias)[:, :, None, None, :]
        pa = F.linear(z3[:, :-1], weight[:, w : 2 * w])
        pa = pa + ell3[:, :-1].to(weight.dtype).unsqueeze(-1) * weight[:, 3 * w]
        pb = F.linear(z3[:, 1:], weight[:, 2 * w : 3 * w])
        pb = pb + ell3[:, 1:].to(weight.dtype).unsqueeze(-1) * weight[:, 3 * w + 1]
        pre = base + pa.unsqueeze(3) + pb.unsqueeze(2)  # [R, S-1, K_a, K_b, H]
        delta = self.mlp2(F.gelu(pre)).squeeze(-1).float()
        return cand3, ell3[:, 1:].unsqueeze(2) + delta

    def select(
        self, x: torch.Tensor, lm_head_weight: torch.Tensor, embed_weight: torch.Tensor
    ) -> torch.Tensor:
        """The walk: slot 0 takes its own argmax, slot ``j`` the best candidate of its table row selected
        by the predecessor the walk chose. ``x`` ``[R, S, D]`` -> draft tokens ``[R, S]``."""
        cand, table = self.tables(x, lm_head_weight, embed_weight)
        k = self.topk
        idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        tokens = [cand[:, 0, 0]]
        for j in range(1, x.shape[1]):
            row = (
                table[:, j - 1]
                .gather(1, idx[:, None, None].expand(-1, 1, k))
                .squeeze(1)
            )
            idx = row.argmax(dim=-1)
            tokens.append(cand[:, j].gather(-1, idx[:, None]).squeeze(-1))
        return torch.stack(tokens, dim=-1)


class DFlashSelectorQwen3Model(DFlashQwen3Model):
    """DFlash Qwen3 backbone + scratch-token inputs + the selector head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        self.num_scratch_tokens = int(getattr(config, "num_scratch_tokens", 0) or 0)
        if self.num_scratch_tokens > 0:
            self.scratch_embed = nn.Parameter(
                torch.zeros(
                    self.num_scratch_tokens,
                    config.hidden_size,
                    dtype=vllm_config.model_config.dtype,
                ),
                requires_grad=False,
            )
        self.selector = DFlashSelectorHead(
            config.hidden_size,
            topk=getattr(config, "selector_topk", 16),
            width=getattr(config, "selector_width", 128),
            mlp_hidden=getattr(config, "selector_hidden", 256),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
            if self.num_scratch_tokens > 0:
                # The query tokens are laid out group by group, the slot first and its scratch
                # tokens after it; a scratch token adds its own learned vector to its slot's input.
                group = 1 + self.num_scratch_tokens
                in_group = torch.arange(
                    input_embeds.shape[0], device=input_embeds.device
                )
                in_group = in_group % group
                table = F.pad(
                    self.scratch_embed, (0, 0, 1, 0)
                )  # row 0 = the slot token: none
                input_embeds = input_embeds + table[in_group].to(input_embeds.dtype)

        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DFlashSelectorQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlashSelectorQwen3Model

    def select_path(self, slot_hidden: torch.Tensor) -> torch.Tensor:
        """Draft tokens ``[R, S]`` from the slots' final hidden states ``[R, S, D]``."""
        return self.model.selector.select(
            slot_hidden, self.lm_head.weight, self.model.embed_tokens.weight
        )
