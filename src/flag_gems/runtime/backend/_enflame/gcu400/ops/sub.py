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
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dense binary fast path (identical shape / dtype / contiguous).
#
# Measurements on GCU400 (48 SPs, torch-event) show the body is bandwidth
# bound: a pure copy kernel hits the same number for every dtype and for
# 48..384 CTAs, so the only lever is how much memory traffic each CTA keeps in
# flight. A 64KB-per-operand tile (16384 elements for fp32, 32768 for
# fp16/bf16) with num_warps=2 reaches the bandwidth ceiling on the 1G-element
# shapes, e.g. fp32 sub: 8.02ms (16KB tile, nw1) -> 4.57ms (64KB tile, nw2)
# with aten at 4.66ms; fp16: 4.28ms -> 2.28ms with aten at 2.31ms. The node's
# grid-stride loop absorbs the rounding, so the larger tile also stays at the
# ceiling on the 4.19M..16.7M element shapes (a 64KB tile never loses more than
# 1% there, and 128KB+ tiles do regress for fp32).
# ---------------------------------------------------------------------------
_DENSE_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_GRID_SIZE = 48
# Per-tile byte budget (per operand). 64KB keeps enough outstanding loads to
# saturate the memory system with only 48 CTAs; 128KB regresses for fp32.
_TILE_BYTES = 65536
# The generic path stays balanced (and cheaper at the launch-latency floor) as
# long as its 65536-element tile fits in a single wave of 48 CTAs; only beyond
# that point does it round up to a duplicated grid-stride iteration.
_GENERIC_TILE_ELEMS = 65536
# Below this size one shape-matched tile (single CTA) beats the generic path
# at the launch-latency floor: (64,64) fp32 measures 4.9us with aten, 7.3us
# through the generic tile and 5.0us with the dense body.
_DENSE_FAST_MIN_NUMEL = 4096
# One wave of 48 generic tiles: from here on the dense grid-stride body wins on
# balance alone, so no shape-matched tile is used anymore.
_DENSE_WAVE_NUMEL = _MAX_GRID_SIZE * _GENERIC_TILE_ELEMS + 1


@triton.jit
def _sub_dense_kernel(a_ptr, b_ptr, out_ptr, alpha, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    # Round-robin over tiles so that the last, possibly ragged, wave is at most
    # one tile per CTA instead of a full duplicated iteration.
    for block_start in range(pid * BLOCK, numel, num_progs * BLOCK):
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < numel
        lhs = tl.load(a_ptr + offsets, mask=mask)
        rhs = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, lhs - rhs * alpha, mask=mask)


def _is_dense_pair(A, B):
    return (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and A.dtype == B.dtype
        and A.dtype in _DENSE_FAST_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
        and _DENSE_FAST_MIN_NUMEL <= A.numel() < (1 << 31)
    )


def _launch_dense_sub(A, B, alpha, out):
    numel = A.numel()
    block = _TILE_BYTES // A.element_size()
    num_warps = 2
    if numel < _DENSE_WAVE_NUMEL:
        # Launch-floor sizes: one shape-matched tile is cheaper than a 64KB tile
        # with a nearly empty grid.
        block = max(256, min(block, triton.next_power_of_2(numel)))
        num_warps = 1 if block <= 4096 else 2
    num_tiles = triton.cdiv(numel, block)
    grid = (min(_MAX_GRID_SIZE, num_tiles),)
    with torch_device_fn.device(A.device.index):
        _sub_dense_kernel[grid](
            A, B, out, alpha, numel, BLOCK=block, num_warps=num_warps
        )
    return out


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


# ---------------------------------------------------------------------------
# Complex support (parity with the generic sub, adapted for torch_gcu).
#
# flagtree's triton cannot take complex dtypes (canonicalize_dtype raises
# KeyError: 'complex64') and torch_gcu's topsaten has no complex sub/cast
# kernels (P21), so complex inputs must be decomposed into the float "pair"
# domain before any kernel launch.  The sibling enflame add op already ships
# that decomposition (view_as_real / pair-domain kernel / view_as_complex);
# since A - B * alpha == A + B * (-alpha) for real alpha, the complex path
# simply composes that FlagGems op with a negated alpha.  Real inputs keep
# the original sub_func kernels untouched.
#
# The add import must stay function-local: pointwise_dynamic's codegen
# sweeps every module-level ImportFrom of this file into the generated
# kernel modules, and a relative `.add` import cannot be resolved there
# (the backend is also loadable as a top-level `gcu400` package).
# ---------------------------------------------------------------------------


def _sub_complex(A, B, alpha):
    if isinstance(alpha, complex):
        raise NotImplementedError(
            "sub with complex alpha is not supported on the enflame backend"
        )
    from .add import add as _gems_add

    return _gems_add(A, B, alpha=-alpha)


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME SUB")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            return _sub_complex(A, B, alpha)
        if _is_dense_pair(A, B):
            return _launch_dense_sub(A, B, alpha, torch.empty_like(A))
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        if A.is_complex() or isinstance(B, complex):
            return _sub_complex(A, B, alpha)
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        if B.is_complex() or isinstance(A, complex):
            return _sub_complex(A, B, alpha)
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_ENFLAME SUB_")
    if isinstance(B, torch.Tensor):
        if A.is_complex() or B.is_complex():
            A.copy_(_sub_complex(A, B, alpha))
            return A
        if _is_dense_pair(A, B):
            return _launch_dense_sub(A, B, alpha, A)
        return sub_func(A, B, alpha, out0=A)
    else:
        if A.is_complex() or isinstance(B, complex):
            A.copy_(_sub_complex(A, B, alpha))
            return A
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
