# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator of the AngelSpec E4 / E7 drafters: DFlash backbone, shift-one block, path selector.

The block is DSpark's anchor-as-first layout (``S`` slots, slot ``j`` predicts the ``j + 1``-th token after
the anchor, every slot drafts a token). With scratch tokens (E7, ``num_scratch_tokens = r``) every slot
is followed by ``r`` query tokens that share its position, so a request has ``S * (1 + r)`` query tokens;
only the slot tokens are read out. The draft is not an argmax per slot but the walk of the path selector
over the slots' top-K candidates (``models/qwen3_dflash_selector.py``), which is fixed-shape tensor math
and stays inside the CUDA graph of the draft step. See that module for the model.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


class DFlashSelectorSpeculator(DFlashSpeculator):
    _speculator_name = "DFlashSelector"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf = self.draft_model_config.hf_config
        if not getattr(hf, "shift_one", False):
            raise ValueError(
                "The path-selector drafter is trained in the shift-one layout "
                "(anchor + MASK slots, every slot drafts a token): the draft config "
                "needs shift_one=true."
            )
        num_scratch = int(getattr(hf, "num_scratch_tokens", 0) or 0)
        self.sample_from_anchor = True
        self.group_size = 1 + num_scratch
        self.num_query_per_req = self.num_speculative_steps * self.group_size
        logger.info(
            "%s: %d slots x %d query tokens (%d scratch), top-%d path selector",
            self._speculator_name,
            self.num_speculative_steps,
            self.group_size,
            num_scratch,
            int(getattr(hf, "selector_topk", 16)),
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        # The DSpark loader body is architecture-agnostic (embed / LM-head sharing with the
        # target, non-causal attention config, KV setup), so it is reused as it is.
        return load_dspark_model(target_model, self.vllm_config)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        steps = self.num_speculative_steps
        # (req, slot) order, one row per slot token (the scratch tokens are not read out)
        slot_hidden = head_hidden[self.sample_indices[: num_reqs * steps]]
        slot_hidden = slot_hidden.view(num_reqs, steps, -1)
        self.draft_tokens[:num_reqs, :steps] = self.model.select_path(slot_hidden)
