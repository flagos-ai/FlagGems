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

import triton
import triton.language as tl

from flag_gems.ops.int_mm import (
    _GROUP_M,
    _MAX_DOT_K_WITHOUT_INT32_OVERFLOW,
    _int_mm_impl,
    _launch_dot,
    _launch_outer,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 128
_LARGE_BLOCK_K = 256
_DEEP_BLOCK_K = 512
_DEEP_BLOCK_K_MIN_K = 2048
_DEEP_BLOCK_K_MAX_TILES = 160
_PACKED_MIN_M = 16
_PACKED_N_ALIGNMENT = 64
_PACKED_K_ALIGNMENT = 128


@libentry()
@triton.jit
def _int_mm_packed_mat2_kernel(
    self_ptr,
    packed_mat2_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    self_ptrs = self_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # packed_mat2 is physically contiguous [N, K]. Presenting it directly as
    # logical [K, N] gives HCU's int8 MMAC lowering a K-contiguous B operand.
    mat2_ptrs = packed_mat2_ptr + offs_k[:, None] + offs_n[None, :] * K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        if EVEN_M and EVEN_K:
            self_block = tl.load(self_ptrs)
        else:
            self_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            self_block = tl.load(self_ptrs, mask=self_mask, other=0)

        if EVEN_N and EVEN_K:
            mat2_block = tl.load(mat2_ptrs)
        else:
            mat2_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
            mat2_block = tl.load(mat2_ptrs, mask=mat2_mask, other=0)

        acc = tl.dot(self_block, mat2_block, acc)
        self_ptrs += BLOCK_K * stride_ak
        mat2_ptrs += BLOCK_K

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if EVEN_M and EVEN_N:
        tl.store(out_ptrs, acc)
    else:
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)


class _PackedMat2Cache:
    """A bounded cache for HCU's physical [N, K] mat2 layout."""

    def __init__(self):
        self._entry = None

    @staticmethod
    def _metadata(source):
        return (
            int(source._version),
            tuple(source.shape),
            tuple(source.stride()),
            int(source.storage_offset()),
            source.dtype,
            source.device,
            triton.runtime.driver.active.get_current_stream(source.device.index),
        )

    def get(self, source):
        # Replay must repack the current weight even when Python is not rerun.
        if torch_device_fn.is_current_stream_capturing():
            return source.t().contiguous()

        try:
            metadata = self._metadata(source)
        except (AttributeError, RuntimeError):
            # Inference tensors may not expose a version counter. Repacking is
            # the only safe choice when in-place mutations cannot be detected.
            return source.t().contiguous()

        entry = self._entry
        if entry is not None and entry[0] is source and entry[1] == metadata:
            return entry[2]

        packed = source.t().contiguous()
        # One immutable entry prevents readers from observing a partial update.
        # The strong source reference prevents object/storage reuse; the version
        # tracks writes, and the stream key keeps asynchronous pack/uses ordered.
        self._entry = (source, metadata, packed)
        return packed


_PACKED_MAT2_CACHE = _PackedMat2Cache()


def _launch_packed(self, mat2, out, M, N, K, block_k):
    # Normalize transposed A before the fused MMAC path.
    if self.stride(1) != 1:
        self = self.contiguous()
    packed_mat2 = _PACKED_MAT2_CACHE.get(mat2)
    grid = (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)
    _int_mm_packed_mat2_kernel[grid](
        self,
        packed_mat2,
        out,
        M,
        N,
        K,
        self.stride(0),
        self.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        GROUP_M=_GROUP_M,
        EVEN_M=M % _BLOCK_M == 0,
        EVEN_N=N % _BLOCK_N == 0,
        EVEN_K=K % block_k == 0,
        num_warps=4,
        num_stages=1,
        num_ldmatrixes=0,
        enable_mmacfuse=2,
    )


def _launch(self, mat2, out, M, N, K):
    if K > _MAX_DOT_K_WITHOUT_INT32_OVERFLOW:
        _launch_outer(self, mat2, out, M, N, K)
    elif (
        M >= _PACKED_MIN_M
        and N >= _PACKED_N_ALIGNMENT
        and N % _PACKED_N_ALIGNMENT == 0
        and K % _PACKED_K_ALIGNMENT == 0
    ):
        # Deep K tiles shorten long reductions, but their register footprint
        # hurts occupancy once the output grid has enough parallelism.
        if (
            K >= _DEEP_BLOCK_K_MIN_K
            and K % _DEEP_BLOCK_K == 0
            and triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N)
            <= _DEEP_BLOCK_K_MAX_TILES
        ):
            block_k = _DEEP_BLOCK_K
        else:
            block_k = _LARGE_BLOCK_K if K % _LARGE_BLOCK_K == 0 else _BLOCK_K
        _launch_packed(self, mat2, out, M, N, K, block_k)
    else:
        _launch_dot(
            self,
            mat2,
            out,
            M,
            N,
            K,
            block_m=_BLOCK_M,
            block_n=_BLOCK_N,
            block_k=_BLOCK_K,
            num_warps=4,
            num_stages=1,
            num_ldmatrixes=1,
        )


def int_mm(self, mat2):
    logger.debug("GEMS HYGON INT_MM")
    return _int_mm_impl(self, mat2, _launch)


def int_mm_out(self, mat2, *, out):
    logger.debug("GEMS HYGON INT_MM_OUT")
    return _int_mm_impl(self, mat2, _launch, out=out)


__all__ = ["int_mm", "int_mm_out"]
