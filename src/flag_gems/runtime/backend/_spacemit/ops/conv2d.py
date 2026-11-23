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

"""SpacemiT two-kernel conv2d (im2col, then blocked GEMM).

The previous backend implementation fused im2col and GEMM into one grid and
synchronised the producer blocks with ``smt.global_mbarrier``.  On K1 that
primitive cannot be compiled: the Triton semantic layer raises
``NotImplementedError`` for it, and the prebuilt ``spine-opt`` does not even
know the ``xsmt.global_mbarrier_init`` op (``custom op ... is unknown``).  The
fused im2col additionally wrote ``BLOCK_SIZE_C`` consecutive channels into
consecutive K columns while the flattened weight uses the ``[c][kh][kw]`` order
of ``weight.reshape(oc, -1)``.

This module therefore uses two sequential launches:

1. ``im2col_kernel`` writes ``(N, OH*OW, C*KH*KW)`` columns in weight order;
2. ``bmm_kernel`` reads that buffer and accumulates one output tile over K
   blocks with ``smt.dot``.  No cross-block barrier is required.

Both kernels use a bounded grid with an internal grid-stride loop: the K1
runtime does not reliably execute more than ~512 programs per launch (see the
vendor ``mean_dim`` fix), and the stem of a 256x256 ResNet needs 16384 im2col
rows.  K is processed in blocks which are zero padded by ``boundary_check``, so
no padding lane contributes a value, and the bias is added once after the K
loop.

The kernels are inference-only (fp32, ``groups=1``).  Calls that need autograd
are delegated to the generic FlagGems ``Conv2d`` autograd Function, and
unsupported inference arguments raise ``NotImplementedError`` instead of
silently producing a wrong result.
"""

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# K1 does not reliably execute more than ~512 programs in one launch.
MAX_PROGRAMS = 512


@libentry()
@triton.jit
def im2col_kernel(
    input_ptr,
    buf_ptr,
    C,
    IH,
    IW,
    KH,
    KW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    OH,
    OW,
    ROWS,
    input_stride_n,
    input_stride_h,
    input_stride_w,
    input_stride_c,
    buf_stride_n,
    buf_stride_m,
    buf_stride_k,
    BLOCK_C: tl.constexpr,
):
    """Write one im2col row per (n, oh, ow).

    The buffer column order is ``c * KH * KW + kh * KW + kw`` so that it
    matches ``weight.reshape(OC, -1)``; channel chunks are therefore strided
    by ``KH * KW`` instead of being written as one contiguous K block.  The
    program processes several rows (grid-stride) to respect the K1 launch
    limit.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row in tl.range(pid, ROWS, num_programs):
        n = row // (OH * OW)
        ohow = row % (OH * OW)
        oh = ohow // OW
        ow = ohow % OW
        for c_start in tl.range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            cmask = offs_c < C
            for kh in range(KH):
                h = oh * stride_h - pad_h + kh * dilation_h
                hvalid = (h >= 0) & (h < IH)
                for kw in range(KW):
                    w = ow * stride_w - pad_w + kw * dilation_w
                    valid = hvalid & (w >= 0) & (w < IW)
                    vals = tl.load(
                        input_ptr
                        + n * input_stride_n
                        + h * input_stride_h
                        + w * input_stride_w
                        + offs_c * input_stride_c,
                        mask=cmask & valid,
                        other=0.0,
                    )
                    k_idx = offs_c * (KH * KW) + kh * KW + kw
                    tl.store(
                        buf_ptr
                        + n * buf_stride_n
                        + ohow * buf_stride_m
                        + k_idx * buf_stride_k,
                        vals,
                        mask=cmask,
                    )


@libentry()
@triton.jit
def bmm_kernel(
    weight_ptr,
    bias_ptr,
    buf_ptr,
    output_ptr,
    OC,
    GEMM_M,
    GEMM_K,
    NBATCH,
    weight_stride_oc,
    weight_stride_k,
    buf_stride_n,
    buf_stride_m,
    buf_stride_k,
    output_stride_n,
    output_stride_oc,
    output_stride_m,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_TILES_M: tl.constexpr,
    NUM_TILES_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    """out[oc, m] = sum_k weight[oc, k] * im2col[m, k] (+ bias).

    Each output tile is accumulated over K blocks into the same packed
    accumulator; the bias is added once after the loop.  Every tile load is
    boundary-checked, so a K tail or a non-multiple output channel count is
    zero padded rather than reading into the next row of the contiguous
    buffers.  Tiles are distributed over a bounded grid (grid-stride).
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_tiles = NBATCH * NUM_TILES_M * NUM_TILES_N
    for tile in tl.range(pid, total_tiles, num_programs):
        pid_b = tile // (NUM_TILES_M * NUM_TILES_N)
        local_tile = tile % (NUM_TILES_M * NUM_TILES_N)
        pid_m = local_tile // NUM_TILES_N
        pid_n = local_tile % NUM_TILES_N
        block_m = pid_m * TILE_M
        block_n = pid_n * TILE_N

        if HAS_BIAS:
            offs_n = block_n + tl.arange(0, TILE_N)
            bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < OC, other=0.0)

        acc = tl.zeros((TILE_N, TILE_M), dtype=tl.float32)
        acc = smt.view(acc, (0, 0), (TILE_N, TILE_M), (MICRO_N, MICRO_M))
        num_k = (GEMM_K + BLOCK_K - 1) // BLOCK_K
        for k in tl.range(0, num_k):
            weight_block_ptr = tl.make_block_ptr(
                base=weight_ptr,
                shape=(OC, GEMM_K),
                strides=(weight_stride_oc, weight_stride_k),
                offsets=(block_n, k * BLOCK_K),
                block_shape=(TILE_N, BLOCK_K),
                order=(1, 0),
            )
            weight_tile = tl.load(weight_block_ptr, boundary_check=(0, 1))
            weight_packed = smt.view(
                weight_tile, (0, 0), (TILE_N, BLOCK_K), (MICRO_N, MICRO_K)
            )

            im2col_block_ptr = tl.make_block_ptr(
                base=buf_ptr,
                shape=(1, GEMM_M, GEMM_K),
                strides=(buf_stride_n, buf_stride_m, buf_stride_k),
                offsets=(pid_b, block_m, k * BLOCK_K),
                block_shape=(1, TILE_M, BLOCK_K),
                order=(2, 1, 0),
            )
            im2col_tile = tl.load(im2col_block_ptr, boundary_check=(0, 1, 2))
            im2col_tile = tl.trans(tl.reshape(im2col_tile, (TILE_M, BLOCK_K)))
            im2col_packed = smt.view(
                im2col_tile, (0, 0), (BLOCK_K, TILE_M), (MICRO_K, MICRO_M)
            )
            acc += smt.dot(weight_packed, im2col_packed)

        acc = smt.view(acc, (0, 0), (TILE_N, TILE_M), (1, 1))
        if HAS_BIAS:
            acc += bias_vals[:, None]
        acc = acc.to(output_ptr.dtype.element_ty)
        output_block_ptr = tl.make_block_ptr(
            base=output_ptr + pid_b * output_stride_n,
            shape=(OC, GEMM_M),
            strides=(output_stride_oc, output_stride_m),
            offsets=(block_n, block_m),
            block_shape=(TILE_N, TILE_M),
            order=(1, 0),
        )
        tl.store(output_block_ptr, acc, boundary_check=(0, 1))


def _pair(value, name):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise NotImplementedError(
                f"GEMS_SPACEMIT CONV2D: only int or 2-element {name} is supported"
            )
        return int(value[0]), int(value[1])
    raise NotImplementedError(
        f"GEMS_SPACEMIT CONV2D: unsupported {name} value {value!r}"
    )


def _needs_autograd(input, weight, bias):
    if not torch.is_grad_enabled():
        return False
    tensors = (input, weight, bias)
    return any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV2D via im2col + bmm kernels")

    if _needs_autograd(input, weight, bias):
        # The two kernels above only implement the forward pass.  Keep the
        # training contract of the generic operator (and of the upstream
        # ``test_conv2d`` gradient assertions) by using its autograd Function.
        from flag_gems.ops.conv2d import Conv2d as _GenericConv2d

        return _GenericConv2d.apply(
            input, weight, bias, stride, padding, dilation, groups
        )

    if not isinstance(padding, (int, tuple, list)):
        raise NotImplementedError(
            f"GEMS_SPACEMIT CONV2D: string padding {padding!r} is not supported"
        )
    if groups != 1:
        raise NotImplementedError(
            f"GEMS_SPACEMIT CONV2D: groups={groups} is not supported (only groups=1)"
        )
    if input.dim() != 4 or weight.dim() != 4:
        raise NotImplementedError("GEMS_SPACEMIT CONV2D: only 4D input/weight")
    if input.dtype != torch.float32 or weight.dtype != torch.float32:
        raise NotImplementedError(
            "GEMS_SPACEMIT CONV2D: only float32 input/weight"
        )
    if bias is not None and bias.dtype != torch.float32:
        raise NotImplementedError("GEMS_SPACEMIT CONV2D: only float32 bias")
    if input.device.type != "cpu":
        raise NotImplementedError(
            f"GEMS_SPACEMIT CONV2D: unsupported device {input.device}"
        )

    N, C, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    if weight_c != C:
        raise ValueError(
            f"expected weight.shape[1] == input.shape[1], got {weight_c} vs {C}"
        )
    if bias is not None and bias.numel() != OC:
        raise ValueError(f"expected {OC} bias values, got {bias.numel()}")

    str_h, str_w = _pair(stride, "stride")
    pad_h, pad_w = _pair(padding, "padding")
    dil_h, dil_w = _pair(dilation, "dilation")
    if min(str_h, str_w) < 1 or min(dil_h, dil_w) < 1 or min(pad_h, pad_w) < 0:
        raise ValueError(
            f"invalid stride={stride}, padding={padding}, dilation={dilation}"
        )

    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // str_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // str_w + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(
            f"calculated output size {(OH, OW)} is not positive for shape "
            f"{(N, C, H, W)}"
        )

    output = torch.empty((N, OC, OH, OW), dtype=input.dtype, device=input.device)
    if N == 0 or OC == 0 or OH * OW == 0:
        return output

    GEMM_M = OH * OW
    GEMM_K = C * KH * KW
    im2col_buf = torch.empty(
        (N, GEMM_M, GEMM_K), dtype=input.dtype, device=input.device
    )
    input_nhwc = input.permute(0, 2, 3, 1).contiguous()
    weight_flat = weight.reshape(OC, -1).contiguous()

    BLOCK_C = min(triton.next_power_of_2(C), 128)
    TILE_M = 32 if GEMM_M >= 32 else 8
    TILE_N = max(16, min(64, triton.next_power_of_2(OC)))
    # At least 32 columns: a single 8-wide micro-K tile hits a broken K1
    # lowering (NaN output), while >=4 tiles are correct.
    BLOCK_K = max(32, min(256, triton.next_power_of_2(GEMM_K)))
    MICRO_M = 8
    MICRO_K = 8
    MICRO_N = 16

    rows = N * GEMM_M
    num_tiles_n = triton.cdiv(OC, TILE_N)
    num_tiles_m = triton.cdiv(GEMM_M, TILE_M)
    total_tiles = N * num_tiles_m * num_tiles_n
    bias_ptr = (
        bias.contiguous()
        if bias is not None
        else torch.empty(0, device=input.device, dtype=input.dtype)
    )
    output_3d = output.view(N, OC, GEMM_M)

    with torch_device_fn.device(input.device):
        im2col_kernel[(min(rows, MAX_PROGRAMS),)](
            input_nhwc,
            im2col_buf,
            C,
            H,
            W,
            KH,
            KW,
            str_h,
            str_w,
            pad_h,
            pad_w,
            dil_h,
            dil_w,
            OH,
            OW,
            rows,
            input_nhwc.stride(0),
            input_nhwc.stride(1),
            input_nhwc.stride(2),
            input_nhwc.stride(3),
            im2col_buf.stride(0),
            im2col_buf.stride(1),
            im2col_buf.stride(2),
            BLOCK_C=BLOCK_C,
        )
        bmm_kernel[(min(total_tiles, MAX_PROGRAMS),)](
            weight_flat,
            bias_ptr,
            im2col_buf,
            output_3d,
            OC,
            GEMM_M,
            GEMM_K,
            N,
            weight_flat.stride(0),
            weight_flat.stride(1),
            im2col_buf.stride(0),
            im2col_buf.stride(1),
            im2col_buf.stride(2),
            output_3d.stride(0),
            output_3d.stride(1),
            output_3d.stride(2),
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            BLOCK_K=BLOCK_K,
            NUM_TILES_M=num_tiles_m,
            NUM_TILES_N=num_tiles_n,
            HAS_BIAS=bias is not None,
            MICRO_M=MICRO_M,
            MICRO_K=MICRO_K,
            MICRO_N=MICRO_N,
        )

    return output