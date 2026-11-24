# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

import flag_gems
from flag_gems.runtime.backend._enflame.gcu400.ops.embedding import (
    _NATIVE_BACKWARD_STATE,
    _embedding_backward_atomic,
)

logger = logging.getLogger(__name__)


def embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    """Dense backward of Embedding: accumulate the grad_output rows into grad_weight.

    The previous Triton version used ``grad_weight_ptr + idx * EMBED_DIM + offs_d``
    for the atomic_add, i.e. "contiguous access after a runtime scalar offset",
    which the gcu400 Triton backend does not support (the benchmark aborts with
    SIGABRT and all 27 test cases mismatch numerically).

    The main path forwards to the native aten ``embedding_dense_backward``; the
    Triton atomic of this backend is ~80x slower than aten, so the atomic fallback
    of the embedding module is reused only when flag_gems has already taken over
    that aten op (enable scenario) and aten can no longer be entered.
    """
    logger.debug("GEMS_ENFLAME EMBEDDING_DENSE_BACKWARD")
    assert indices.dtype in (
        torch.int32,
        torch.int64,
    ), "Indices must be int32 or int64."
    if (
        grad_output.device.type != flag_gems.device
        or indices.device.type != flag_gems.device
        or grad_output.device != indices.device
    ):
        raise ValueError(
            f"Inputs must be {flag_gems.device} tensors on the same device."
        )

    assert (
        grad_output.dim() >= 2
    ), "grad_output must have embedding dimension as the last dim."

    grad_output = grad_output.contiguous()
    indices = indices.contiguous()
    # topsaten only accepts int32 index tensors: an int64 index tensor aborts
    # inside the native kernel (TOPSATEN_STATUS_BAD_PARAM, "index tensor must be
    # of int32 data type") instead of raising, which kills the process. Cast here
    # so the int64-enabled default configuration stays usable.
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    if getattr(_NATIVE_BACKWARD_STATE, "forwarding", False):
        return _embedding_backward_atomic(
            grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
        )

    # aten uses -1 to mean "no padding_idx".
    normalized_padding_idx = -1 if padding_idx is None else padding_idx
    _NATIVE_BACKWARD_STATE.forwarding = True
    try:
        python_key = torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
        with torch._C._ExcludeDispatchKeyGuard(python_key):
            return torch.ops.aten.embedding_dense_backward(
                grad_output,
                indices,
                num_weights,
                normalized_padding_idx,
                scale_grad_by_freq,
            )
    finally:
        _NATIVE_BACKWARD_STATE.forwarding = False
