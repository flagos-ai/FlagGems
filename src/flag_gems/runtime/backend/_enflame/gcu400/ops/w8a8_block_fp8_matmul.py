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

"""Enflame GCU400 block-wise W8A8 FP8 GEMM.

``out = (A @ B^T) * scale_a * scale_b`` with A ``[M, K]`` FP8, B ``[N, K]``
FP8 (K is the contiguous axis on both sides), ``scale_a`` of shape
``[M, ceil(K / block_k)]`` and ``scale_b`` of shape
``[ceil(N / block_n), ceil(K / block_k)]``.

Design notes (measured on GCU400):

* The generic kernel walks B with the N axis as the fastest axis, which turns
  the B tile into a per-element gather.  Here B is loaded as a contiguous
  ``[TILE_N, TILE_K]`` tile and transposed into the dot, so both operands come
  from row-contiguous tile DMA.
* ``tl.dot`` keeps FP8 operands on a slow (emulated) path; casting to FP16 --
  exact for every finite E4M3/E5M2 value -- is ~23x faster.  The transpose must
  be applied *after* the cast: transposing an FP8 tile lowers to a
  ``tops_transpose_async_2_fp8e4m3fn`` call that fails to link.
* Row offsets must stay affine.  Clamping them (``tl.minimum(offs, M - 1)``) to
  drop the load mask makes the base of every tile DMA data dependent and costs
  ~2.6x; a plain row mask over affine offsets is the fast form.
* A single quantized k block must not straddle two tiles, so ``TILE_K`` divides
  ``block_k`` and ``TILE_N`` divides ``block_n``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Largest k tile that still fits inside one quantization block for the
# block_k = 128 configurations used by the W8A8 block FP8 checkpoints.
_MAX_TILE_K = 128
# GCU400 launcher limit on grid.x.
_MAX_GRID_X = 65535


@libentry()
@triton.jit
def _w8a8_block_fp8_matmul_kernel(
    A,
    B,
    C,
    AS,
    BS,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    BS_SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, TILE_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    as_ptrs = AS + offs_m * stride_asm
    if BS_SCALAR:
        bs_ptr = BS + (pid_n * TILE_N // GROUP_N) * stride_bsn
    else:
        bs_ptrs = BS + (offs_n // GROUP_N) * stride_bsn

    scale_k_stride: tl.constexpr = GROUP_K // TILE_K

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for kt in range(tl.cdiv(K, TILE_K)):
        k_offs = kt * TILE_K + offs_k
        k_mask = k_offs < K
        # Addresses are rebuilt from the induction variable each iteration: a
        # loop-carried pointer vector is materialized as a scalar loop here.
        a = tl.load(
            A + offs_m[:, None] * stride_am + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B + offs_n[:, None] * stride_bn + k_offs[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        partial = tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))

        group = kt // scale_k_stride
        a_scale = tl.load(as_ptrs + group * stride_ask, mask=m_mask, other=0.0)
        if BS_SCALAR:
            b_scale = tl.load(bs_ptr + group * stride_bsk)
            acc += partial * (a_scale * b_scale)[:, None]
        else:
            b_scale = tl.load(bs_ptrs + group * stride_bsk)
            acc += partial * a_scale[:, None] * b_scale[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _pick_tile_k(group_k):
    for tile_k in (_MAX_TILE_K, 64, 32, 16):
        if group_k % tile_k == 0:
            return tile_k
    return group_k


def _pick_config(m, n, k, group_n, group_k):
    """Tile/launch configuration for one problem size.

    ``TILE_N`` is capped by ``group_n`` so an n tile never straddles two scale
    blocks.  One warp per CTA is the measured best point on GCU400 for the
    wide-N shapes (the grid sits above the 48 CTA wave for every realistic N).
    """
    tile_k = _pick_tile_k(group_k)
    if group_n % 128 == 0:
        tile_n = 128
    elif group_n % 64 == 0:
        tile_n = 64
    else:
        tile_n = group_n
    if m <= 16:
        tile_m = 16
    elif m <= 64:
        tile_m = 64
    else:
        tile_m = 128
    return tile_m, tile_n, tile_k, 1, 3


def _launch(a, b, c, as_, bs, m, n, k, group_n, group_k):
    tile_m, tile_n, tile_k, warps, stages = _pick_config(m, n, k, group_n, group_k)
    grid = triton.cdiv(m, tile_m) * triton.cdiv(n, tile_n)
    if grid > _MAX_GRID_X:
        raise ValueError(f"w8a8_block_fp8_matmul grid {grid} exceeds {_MAX_GRID_X}")
    if not as_.is_contiguous():
        as_ = as_.reshape(m, as_.shape[-1]).contiguous()
    stride_asm, stride_ask = as_.stride(-2), as_.stride(-1)
    _w8a8_block_fp8_matmul_kernel[(grid,)](
        a,
        b,
        c,
        as_,
        bs,
        m,
        n,
        k,
        a.stride(-2),
        b.stride(0),
        c.stride(-2),
        stride_asm,
        stride_ask,
        bs.stride(0),
        bs.stride(1),
        GROUP_N=group_n,
        GROUP_K=group_k,
        TILE_M=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        BS_SCALAR=group_n % tile_n == 0,
        num_warps=warps,
        num_stages=stages,
    )
    return c


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise W8A8 FP8 matmul with per-block dequantization scales.

    ``A`` is FP8 ``[..., K]`` (contiguous), ``B`` is FP8 ``[N, K]``, ``As`` is
    FP32 ``[..., ceil(K / block_k)]``, ``Bs`` is FP32
    ``[ceil(N / block_n), ceil(K / block_k)]``; the result is
    ``[..., N]`` in ``output_dtype``.
    """
    block_n, block_k = int(block_size[0]), int(block_size[1])
    k = A.shape[-1]
    n = B.shape[0]
    m = A.numel() // k
    c = A.new_empty(A.shape[:-1] + (n,), dtype=output_dtype)
    if m == 0 or n == 0:
        return c
    with torch_device_fn.device(A.device):
        return _launch(A, B, c, As, Bs, m, n, k, block_n, block_k)


__all__ = ["w8a8_block_fp8_matmul"]
