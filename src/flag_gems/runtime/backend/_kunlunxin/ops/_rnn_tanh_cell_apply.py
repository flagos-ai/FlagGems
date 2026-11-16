# Copyright 2026, The FlagOS Contributors.
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
#
# ``rnn_tanh_cell`` activation kernel for the Kunlunxin override, kept in its
# own module on purpose: a kernel containing ``tl_extra_shim.tanh`` defined in
# the operator module that also holds ``tl.dot`` kernels aborts a *cold* XPU
# compile with ``KeyError: 'tt.tanh'`` once a dot kernel has been built in the
# same process, while the identical kernel imported from a sibling module
# compiles cleanly.  See
# ``wiki/patterns/xpu3-cold-compile-sigmoid-lut-module.md``.
#
# ``pre`` holds ``round(g_i + g_h)`` already rounded to the operator dtype; the
# only work left is ``out = tanh(pre)`` in fp32 with one final store rounding,
# exactly like the generic ``_rnn_tanh_cell_kernel`` tail.
#
# The flat tile follows the ``_mkldnn_rnn_layer_gate`` lesson: sub-warp blocks
# are widened by this backend, so the masked variant clamps the lane index
# (loads always in bounds, no reliance on masked-load ``other``) and keeps an
# explicit store mask.  ``pre``/``out`` share the flat ``(batch, hidden)``
# layout; the fast-path ``pre`` may carry trailing padded rows, which are simply
# never read.

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

_tanh = tl_extra_shim.tanh


@triton.jit
def rnn_tanh_cell_apply_kernel(
    pre_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    raw = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lane_mask = raw < N
    offs = tl.minimum(raw, N - 1)
    value = tl.load(pre_ptr + offs).to(tl.float32)
    tl.store(
        out_ptr + offs,
        _tanh(value).to(out_ptr.dtype.element_ty),
        mask=lane_mask,
    )


@triton.jit
def rnn_tanh_cell_apply_kernel_full(
    pre_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(pre_ptr + offs).to(tl.float32)
    tl.store(
        out_ptr + offs,
        _tanh(value).to(out_ptr.dtype.element_ty),
    )
