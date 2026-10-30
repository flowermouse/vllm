# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DFlash draft with AngelSpec's conditional refinement layer (CRL).

AngelSpec (Scratch-Tokens-Impl.md section 27) trains the DFlash block (``[anchor, MASK x 15]``; row ``k``
predicts the ``k``-th token after the anchor) with a refinement layer on top of the last decoder layer
instead of DFlash 2's path selector. For row ``k`` and a predecessor token ``a`` (row 1: the anchor token;
rows 2..15: the top ``crl_pred_topk`` candidates of row ``k - 1``)::

    r   = h_k + W_e RMSNorm_e(E_in[a])                    # h_k: the last layer's output before the final norm
    r'  = r + o_proj(Attn(RoPE_{p_k}(q_norm(q_proj RMSNorm_1(r))), K, V))   # the last layer's own K/V
    r'' = r' + SwiGLU(RMSNorm_2(r'))
    S_k(a, b) = final_norm(r'') . W_out[b]                # b over the row's top ``selector_top_k`` candidates

The refinement queries read the last layer's paged K/V cache (context + the 16 block rows, non-causal, exactly
what row ``k`` itself sees) and write nothing, so the draft step is the stock DFlash step plus one extra
attention call. The ``K x K`` score tables go through the DFlash 2 walk kernel: a predecessor outside the
first ``crl_pred_topk`` candidates falls back to the row's own logits (= its own argmax).

The draft proposes a deterministic path, so it is meant for greedy verification (temperature 0), like E4 / E7.
The convolutions of DFlash 2 are optional (``dflash_config.conv_kernel_size``).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .qwen3_dflash2 import DFlash2Qwen3DecoderLayer


class CRLayer(nn.Module):
    """The refinement layer. Parameter names are AngelSpec's (``layers.{L-1}.crl.*``)."""

    def __init__(self, config, draft_config: dict) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or d // self.num_heads
        eps = config.rms_norm_eps
        hidden = int(draft_config["crl_mlp_hidden"])
        self.pred_topk = int(draft_config["crl_pred_topk"])
        self.w_e = nn.Linear(d, d, bias=False)
        self.norm_e = RMSNorm(d, eps=eps)
        self.norm1 = RMSNorm(d, eps=eps)
        self.q_proj = nn.Linear(d, self.num_heads * self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d, bias=False)
        self.norm2 = RMSNorm(d, eps=eps)
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, h: torch.Tensor, emb: torch.Tensor, rope, attend) -> torch.Tensor:
        """``h`` / ``emb`` ``[N, D]`` (the row's last-layer output and the predecessor's input embedding) -> ``r''`` ``[N, D]``.
        ``rope(q)`` rotates ``q`` ``[N, H * Dh]`` in place at the rows' positions, ``attend(q)`` ``[N, H, Dh]`` -> ``[N, H, Dh]``."""
        r = h + self.w_e(self.norm_e(emb))
        q = self.q_proj(self.norm1(r))
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).reshape(q.shape)
        rope(q)
        o = attend(q.view(-1, self.num_heads, self.head_dim))
        r = r + self.o_proj(o.reshape(o.shape[0], -1))
        x = self.norm2(r)
        return r + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _NormWithPre(RMSNorm):
    """The draft's final norm that also hands out the sum it normalised: ``[norm(x + residual) | x + residual]``.

    The refinement layer needs the last layer's output before the final norm; returning it next to the normed
    hidden state keeps the compiled model a single-output function (and the CUDA graph's output buffer one tensor).
    """

    def forward(self, x: torch.Tensor, residual: torch.Tensor):  # type: ignore[override]
        pre = x + residual  # (before the fused add-norm, which may update `residual` in place)
        out, res = super().forward(x, residual)
        return torch.cat([out, pre], dim=-1), res


class DFlashCRLQwen3Model(DFlashQwen3Model):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        config = spec.draft_model_config.hf_config
        draft_config = config.dflash_config
        if int(draft_config.get("conv_kernel_size", 0) or 0) > 0:
            self.decoder_layer_cls = DFlash2Qwen3DecoderLayer
        if int(getattr(config, "num_scratch_tokens", 0) or 0) > 0:
            raise ValueError("The vLLM CRL draft does not support scratch tokens.")
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        self.candidate_topk = int(draft_config.get("selector_top_k", 16))
        self.crl = CRLayer(config, draft_config)
        if self.crl.pred_topk > self.candidate_topk:
            raise ValueError("crl_pred_topk must not exceed selector_top_k.")
        # same parameter name / path as the stock final norm, so the checkpoint's ``norm.weight`` still loads
        norm = _NormWithPre(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = norm


class DFlashCRLQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlashCRLQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.draft_id_to_target_id is not None:
            raise ValueError("The CRL draft needs the full target vocabulary.")
        self.candidate_logits_processor = LogitsProcessor(
            vllm_config.model_config.get_vocab_size()
        )

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-K candidate ids and their logits; ``hidden_states`` is ``[N, 2 D]`` = ``[normed | pre-norm]`` (the model's output)."""
        hidden = hidden_states[..., : self.config.hidden_size]
        return self.candidate_logits_processor.get_top_k_tokens(
            self.lm_head, hidden, self.model.candidate_topk
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        last = self.config.num_hidden_layers - 1
        prefix = f"layers.{last}.crl."
        crl_weights: list[tuple[str, torch.Tensor]] = []
        seen: set[str] = set()

        def backbone_weights():
            for name, value in weights:
                bare = name.removeprefix("model.")
                if bare.startswith(prefix):
                    # the refinement layer is a module of its own: keep it away from the backbone's stacked
                    # q/k/v and gate/up name mapping
                    crl_weights.append((bare.removeprefix(prefix), value))
                    seen.add(bare.removeprefix(prefix))
                else:
                    yield name, value

        super().load_weights(backbone_weights())
        expected = {n for n, _ in self.model.crl.named_parameters()}
        if seen != expected:
            raise ValueError(
                f"CRL checkpoint coverage mismatch: missing={sorted(expected - seen)}, "
                f"unexpected={sorted(seen - expected)}"
            )
        AutoWeightsLoader(self.model.crl).load_weights(crl_weights)


EntryClass = DFlashCRLQwen3ForCausalLM
