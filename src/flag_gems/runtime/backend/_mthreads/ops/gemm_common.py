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

"""Small kernels shared by the MThreads GEMM-family operators."""

import triton
import triton.language as tl

from flag_gems.utils import libentry


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def small_m_nt_gemv_kernel(
    a,
    b,
    bias,
    out,
    alpha,
    beta,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_bias_b,
    stride_bias_m,
    stride_bias_n,
    stride_out_b,
    stride_out_m,
    stride_out_n,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FUSE_BIAS: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    """Compute one or two rows against a physical ``[B, N, K]`` tensor."""

    pid_b = tl.program_id(1)
    if BLOCK_N == 1:
        offset_n = tl.program_id(0)
        accumulator0 = tl.zeros((), dtype=tl.float32)
        accumulator1 = tl.zeros((), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < K
            weight = tl.load(
                b + pid_b * stride_bb + offsets_k * stride_bk + offset_n * stride_bn,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            vector0 = tl.load(
                a + pid_b * stride_ab + offsets_k * stride_ak,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            accumulator0 += tl.sum(weight * vector0, axis=0)
            if M == 2:
                vector1 = tl.load(
                    a + pid_b * stride_ab + stride_am + offsets_k * stride_ak,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                accumulator1 += tl.sum(weight * vector1, axis=0)

        if FUSE_BIAS:
            if BETA_ZERO:
                output0 = accumulator0 * alpha
            else:
                bias0 = tl.load(
                    bias + pid_b * stride_bias_b + offset_n * stride_bias_n
                ).to(tl.float32)
                output0 = accumulator0 * alpha + bias0 * beta
        else:
            output0 = accumulator0
        tl.store(
            out + pid_b * stride_out_b + offset_n * stride_out_n,
            output0,
        )
        if M == 2:
            if FUSE_BIAS:
                if BETA_ZERO:
                    output1 = accumulator1 * alpha
                else:
                    bias1 = tl.load(
                        bias
                        + pid_b * stride_bias_b
                        + stride_bias_m
                        + offset_n * stride_bias_n
                    ).to(tl.float32)
                    output1 = accumulator1 * alpha + bias1 * beta
            else:
                output1 = accumulator1
            tl.store(
                out + pid_b * stride_out_b + stride_out_m + offset_n * stride_out_n,
                output1,
            )
    else:
        offsets_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N
        accumulator0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        accumulator1 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offsets_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offsets_k < K
            weight = tl.load(
                b
                + pid_b * stride_bb
                + offsets_k[None, :] * stride_bk
                + offsets_n[:, None] * stride_bn,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            vector0 = tl.load(
                a + pid_b * stride_ab + offsets_k * stride_ak,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            accumulator0 += tl.sum(weight * vector0[None, :], axis=1)
            if M == 2:
                vector1 = tl.load(
                    a + pid_b * stride_ab + stride_am + offsets_k * stride_ak,
                    mask=mask_k,
                    other=0.0,
                ).to(tl.float32)
                accumulator1 += tl.sum(weight * vector1[None, :], axis=1)

        if FUSE_BIAS:
            if BETA_ZERO:
                output0 = accumulator0 * alpha
            else:
                bias0 = tl.load(
                    bias + pid_b * stride_bias_b + offsets_n * stride_bias_n,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                output0 = accumulator0 * alpha + bias0 * beta
        else:
            output0 = accumulator0

        tl.store(
            out + pid_b * stride_out_b + offsets_n * stride_out_n,
            output0,
            mask=mask_n,
        )
        if M == 2:
            if FUSE_BIAS:
                if BETA_ZERO:
                    output1 = accumulator1 * alpha
                else:
                    bias1 = tl.load(
                        bias
                        + pid_b * stride_bias_b
                        + stride_bias_m
                        + offsets_n * stride_bias_n,
                        mask=mask_n,
                        other=0.0,
                    ).to(tl.float32)
                    output1 = accumulator1 * alpha + bias1 * beta
            else:
                output1 = accumulator1
            tl.store(
                out + pid_b * stride_out_b + stride_out_m + offsets_n * stride_out_n,
                output1,
                mask=mask_n,
            )


__all__ = ["small_m_nt_gemv_kernel"]
