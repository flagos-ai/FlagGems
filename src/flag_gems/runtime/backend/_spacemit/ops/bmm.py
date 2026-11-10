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
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm_spacemit"),
    key=["batch", "M", "N", "K"],
    disable_aabs=True,
)
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    batch,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_N: tl.constexpr,
    MICRO_K: tl.constexpr,
    num_ctas: tl.constexpr,
    Bias=None,
    stride_bias_b=0,
    stride_bias_m=0,
):
    # Keep every K-block result in FP32, including for FP16 inputs.
    partial_ptr = smt.alloc(
        (TILE_M // MICRO_M, TILE_N // MICRO_N, MICRO_M, MICRO_N), type=tl.float32
    )
    tiles_m = tl.cdiv(M, TILE_M)
    tiles_n = tl.cdiv(N, TILE_N)
    tiles_per_batch = tiles_m * tiles_n
    for task in tl.range(tl.program_id(0), batch * tiles_per_batch, tl.num_programs(0)):
        pid_b = task // tiles_per_batch
        pid_m = task // tiles_n % tiles_m
        pid_n = task % tiles_n
        acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for ki in tl.range(0, tl.cdiv(K, TILE_K)):
            a_ptr = tl.make_block_ptr(
                base=A + pid_b * stride_ab,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(pid_m * TILE_M, ki * TILE_K),
                block_shape=(TILE_M, TILE_K),
                order=(1, 0),
            )
            b_ptr = tl.make_block_ptr(
                base=B + pid_b * stride_bb,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(ki * TILE_K, pid_n * TILE_N),
                block_shape=(TILE_K, TILE_N),
                order=(1, 0),
            )
            # Dot must see zeros beyond the logical K dimension.
            a_tile = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
            b_tile = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
            a = smt.view(a_tile, (0, 0), (TILE_M, TILE_K), (MICRO_M, MICRO_K))
            b = smt.view(b_tile, (0, 0), (TILE_K, TILE_N), (MICRO_K, MICRO_N))
            partial = smt.descriptor_load(partial_ptr, (0, 0, 0, 0))
            smt.dot(a, b, partial)
            acc += smt.view(partial, (0, 0), (TILE_M, TILE_N), (1, 1))
        if Bias is not None:
            rows = pid_m * TILE_M + tl.arange(0, TILE_M)
            bias = tl.load(
                Bias + pid_b * stride_bias_b + rows * stride_bias_m,
                rows < M,
                other=0,
            ).to(tl.float32)
            acc += bias[:, None]
        out_ptr = tl.make_block_ptr(
            base=O + pid_b * stride_ob,
            shape=(M, N),
            strides=(stride_om, stride_on),
            offsets=(pid_m * TILE_M, pid_n * TILE_N),
            block_shape=(TILE_M, TILE_N),
            order=(1, 0),
        )
        tl.store(out_ptr, acc.to(O.dtype.element_ty), boundary_check=(0, 1))


def bmm(A, B):
    return bmm_out(A, B)


def bmm_out(A, B, out=None):
    logger.debug("GEMS_SPACEMIT BMM")
    assert A.ndim == B.ndim == 3, "Expected batched matrices"
    batch, M, K = A.shape
    batch_b, K_b, N = B.shape
    assert batch == batch_b, "Batch dim mismatch"
    assert K == K_b, "K dim mismatch"
    assert A.dtype == B.dtype, "Dtype mismatch"
    assert A.device == B.device, "Device mismatch"
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    else:
        assert out.shape == (batch, M, N), "Output shape mismatch"
        assert out.dtype == A.dtype, "Dtype mismatch"
        assert out.device == A.device, "Device mismatch"
    if batch == 0 or M == 0 or N == 0:
        return out
    if K == 0:
        # The overridden zero_ assumes contiguous output; copy handles strides.
        out.copy_(torch.zeros(out.shape, dtype=out.dtype, device=out.device))
        return out
    if A.dtype not in (torch.float16, torch.float32):
        from flag_gems.ops.bmm import bmm_out as general_bmm_out

        return general_bmm_out(A, B, out)

    # Packing supports contiguous or transposed matrix dimensions.
    stride_a, stride_b = A.stride(), B.stride()
    if stride_a[1] != 1 and stride_a[2] != 1:
        A = A.contiguous()
        stride_a = A.stride()
    if stride_b[1] != 1 and stride_b[2] != 1:
        B = B.contiguous()
        stride_b = B.stride()

    def grid(meta):
        tiles = batch * triton.cdiv(M, meta["TILE_M"]) * triton.cdiv(N, meta["TILE_N"])
        # K3 launches above 512 programs are unsafe; each program can do more tiles.
        return (min(tiles, meta["num_ctas"], 512),)

    bmm_kernel[grid](
        A,
        B,
        out,
        batch,
        M,
        N,
        K,
        *stride_a,
        *stride_b,
        *out.stride(),
    )
    return out
