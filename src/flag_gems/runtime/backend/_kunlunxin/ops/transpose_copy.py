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

"""Kunlunxin (P800) override for ``aten::transpose_copy.int``.

The generic implementation transposes with two dedicated tiled kernels
(``_transpose_copy_tiled_kernel`` / ``_complex_transpose_tiled_kernel``) that
both build a 2D masked tile and call ``tl.trans``. On the TritonXPU backend that
pattern fails to compile (``ConvertTritonToTritonXPU`` pipeline aborts), which
made every genuine-transpose case (real dtypes, float8 byte views, and complex)
crash at build time.

The generic 1D word-gather kernel (``_complex_copy_kernel``) does compile and run
correctly on XPU -- the same-dim / strided copy cases already pass through it.
This override routes *every* transpose_copy case through that single gather
kernel by reinterpreting the (logically transposed) input as a strided buffer of
integer words and copying it into a contiguous output. No ATen / native compute
fallback is used; the transpose is a pure strided bit-copy.
"""

import logging

import torch
import triton

from flag_gems.ops.transpose_copy import (
    _complex_copy_kernel,
    _has_lazy_metadata,
    _normalize_dim,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.tensor_wrapper import StridedBuffer

logger = logging.getLogger(__name__)

_BLOCK_SIZE = 512
_WORD_DTYPE = {
    8: torch.uint8,
    16: torch.uint16,
    32: torch.uint32,
    64: torch.uint64,
}


def _word_layout(input: torch.Tensor):
    """Pick the integer word width used to move the bits without FP/complex math.

    Mirrors the generic ``_launch_complex_copy`` word selection so the complex
    sign-flip logic in ``_complex_copy_kernel`` keeps working, and extends it to
    1-byte elements (float8 / int8 / uint8 / bool) which the generic path only
    ever reached through ``_view_as_uint8``.
    """
    element_size = input.element_size()
    lazy = _has_lazy_metadata(input)
    word_bits = 16 if element_size == 4 else 32
    if not lazy and element_size <= 8:
        word_bits = element_size * 8
    words = element_size * 8 // word_bits
    return word_bits, words, _WORD_DTYPE[word_bits]


def _launch_word_gather(
    input: torch.Tensor, out: torch.Tensor, dim0: int, dim1: int
) -> torch.Tensor:
    view = input if dim0 == dim1 else input.transpose(dim0, dim1)

    word_bits, words, word_dtype = _word_layout(input)
    sign_mask = (1 << (word_bits - 1)) if _has_lazy_metadata(input) else 0

    kernel_input = StridedBuffer(
        input,
        shape=(*view.shape, words),
        strides=(*(stride * words for stride in view.stride()), 1),
        dtype=word_dtype,
    )
    kernel_out = StridedBuffer(
        out,
        shape=(*out.shape, words),
        strides=(*(stride * words for stride in out.stride()), 1),
        dtype=word_dtype,
    )

    n_words = out.numel() * words
    grid = (triton.cdiv(n_words, _BLOCK_SIZE),)
    with torch_device_fn.device(input.device):
        _complex_copy_kernel[grid](
            kernel_input,
            kernel_out,
            n_words,
            tuple(view.shape),
            tuple(view.stride()),
            words,
            input.is_conj(),
            input.is_neg() if hasattr(input, "is_neg") else False,
            sign_mask,
            _BLOCK_SIZE,
        )
    return out


def transpose_copy(input: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return a contiguous copy with ``dim0`` and ``dim1`` swapped (XPU)."""
    logger.debug("GEMS_KUNLUNXIN TRANSPOSE_COPY")

    normalized_dim0 = _normalize_dim(dim0, input.ndim)
    normalized_dim1 = _normalize_dim(dim1, input.ndim)

    out_shape = list(input.shape)
    if input.ndim > 0:
        out_shape[normalized_dim0], out_shape[normalized_dim1] = (
            out_shape[normalized_dim1],
            out_shape[normalized_dim0],
        )

    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    if input.numel() == 0:
        return out

    return _launch_word_gather(input, out, normalized_dim0, normalized_dim1)
