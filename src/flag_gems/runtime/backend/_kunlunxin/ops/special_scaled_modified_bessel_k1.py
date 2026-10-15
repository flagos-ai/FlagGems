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
# Kunlunxin (XPU) override of special_scaled_modified_bessel_k1 /
# special_scaled_modified_bessel_k1_out (aten::special_scaled_modified_bessel_k1).
#
# Why an override exists at all (performance):
#   the generic `flag_gems/ops/special_scaled_modified_bessel_k1.py` launches
#   its kernel with `grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)`,
#   i.e. a fresh lambda object per call.  On the xpubin backend the launch
#   grid is part of the compilation cache key, so every call becomes a full
#   recompilation: measured ~173-180 ms/call regardless of tensor size
#   (fp32 Gems Speedup ~0.0009-0.56 at core level), which is the
#   KERNEL_OPT_EXPERIENCE #41 fingerprint exactly.  The profiler shows the
#   device kernel itself is only ~8 ms/call at 16.7M elements (99% of device
#   time), so the bottleneck is host-side JIT recompilation, not compute or
#   memory.
#
# Fix (single-variable): vendor raw-pointer kernels with a constant-tuple
# grid plus the sweep-measured `_pick_block` tile buckets of the erfc/ndtr
# pointwise family (the same scheme special_modified_bessel_i0 uses).  The
# math body below is verbatim the generic kernel: three regions
#   small  (|x| <= 2):   K1(x) = log(x/2) * I1(x) + P(x)/x,
#                        I1 via Cephes polynomial, P via Clenshaw Chebyshev
#                        (11 terms, degree 10), scaled by exp(x)
#   mid    (2 < x <= 8): sqrt(x) * scaled_K1(x) as a degree-9 Horner
#                        polynomial in 1/x
#   large  (x > 8):      asymptotic expansion (5 terms) times sqrt(pi/(2x))
# plus the same boundary semantics (x<=0 -> nan, x~0 -> inf).
# `tl.log`/`tl.exp`/`tl.sqrt`/`tl.abs`/`tl.where` are all proven to link on
# XPU in this family (special_modified_bessel_k0 uses the identical
# `isCloseVectorization=True` recipe for its `tl.log`; j0/j1 confirm
# `tl.sqrt`/`tl.exp`).
# No CPU/ATen/native/composite fallback: `torch.empty*` only allocates.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

UNROLL_NUM = 8
BUFFER_SIZE_LIMIT = 2048
IS_CLOSE_MEMORY_ASYNC = False
IS_CLOSE_VECTORIZATION = True


def _pick_block(n_elements):
    # Tile buckets sweep-measured on XPU for the erfc/ndtr pointwise family
    # (identical load/store pattern, see erfc.py / special_ndtr.py).  The
    # masked-memory path costs ~2x on this backend, so an unmasked kernel is
    # used whenever the shape divides the tile exactly.
    if n_elements >= 16_777_216 and n_elements % 65536 == 0:
        return 65536, 16, False
    if n_elements >= 1_048_576 and n_elements % 32768 == 0:
        return 32768, 8, False
    if n_elements >= 65_536 and n_elements % 8192 == 0:
        return 8192, 4, False
    if n_elements <= 65_536:
        return 2048, 4, True
    return 8192, 4, True


@triton.jit
def _k1_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Handle non-positive values
    zero = 0.0
    ax = tl.abs(x_f32)
    is_zero = ax < 1e-30
    is_negative = x_f32 < zero

    # ================================================================
    # Small region: 0 < x <= 2.0
    # scaled_K1(x) = exp(x) * K1(x)
    # K1(x) = log(x/2) * I1(x) + P(x) / x
    # P(x) evaluated via Chebyshev series in w = x - 1 (degree 10, 11 terms).
    # ================================================================

    # --- Step 1: Compute I1(x) via Cephes polynomial (valid for x <= 3.75) ---
    y_i1 = x_f32 / 3.75
    y2_i1 = y_i1 * y_i1
    p_i1 = 0.00032411
    p_i1 = 0.00301532 + y2_i1 * p_i1
    p_i1 = 0.02658733 + y2_i1 * p_i1
    p_i1 = 0.15084934 + y2_i1 * p_i1
    p_i1 = 0.51498869 + y2_i1 * p_i1
    p_i1 = 0.87890594 + y2_i1 * p_i1
    p_i1 = 0.5 + y2_i1 * p_i1
    i1_val = x_f32 * p_i1

    # --- Step 2: Chebyshev series for P(x) at w = x - 1 ---
    # Clenshaw recurrence (3-state, correct update order):
    #   b_2 = b_1; b_1 = b_0; b_0 = c_k + 2*y*b_1 - b_2
    # Result: (b_0 - b_2) / 2
    y_cheb = x_f32 - 1.0
    b0 = 0.0
    b1 = 0.0

    # k=10: T_10 coefficient
    b2 = b1
    b1 = b0
    b0 = -3.662620162661560e-09 + 2.0 * y_cheb * b1 - b2

    # k=9: T_9 coefficient
    b2 = b1
    b1 = b0
    b0 = -5.350617072358099e-08 + 2.0 * y_cheb * b1 - b2

    # k=8: T_8 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.045185163440483e-06 + 2.0 * y_cheb * b1 - b2

    # k=7: T_7 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.201112421320492e-05 + 2.0 * y_cheb * b1 - b2

    # k=6: T_6 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.699042469646379e-04 + 2.0 * y_cheb * b1 - b2

    # k=5: T_5 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.426485944971723e-03 + 2.0 * y_cheb * b1 - b2

    # k=4: T_4 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.225928401715418e-02 + 2.0 * y_cheb * b1 - b2

    # k=3: T_3 coefficient
    b2 = b1
    b1 = b0
    b0 = -6.428271235627786e-02 + 2.0 * y_cheb * b1 - b2

    # k=2: T_2 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.767199860308625e-01 + 2.0 * y_cheb * b1 - b2

    # k=1: T_1 coefficient
    b2 = b1
    b1 = b0
    b0 = -2.944128552518444e-01 + 2.0 * y_cheb * b1 - b2

    # k=0: T_0 coefficient (doubled for Clenshaw halving convention)
    b2 = b1
    b1 = b0
    b0 = 1.658032209918573e00 + 2.0 * y_cheb * b1 - b2

    # Final Chebyshev value: (b_0 - b_2) / 2
    cheb_val = (b0 - b2) / 2.0

    # --- Step 3: Compute K1(x) and scaled_K1(x) ---
    # K1(x) = log(x/2) * I1(x) + cheb_val / x
    # scaled_K1(x) = exp(x) * K1(x)
    log_term = tl.log(tl.maximum(x_f32, 1e-30) / 2.0)
    k1_small = log_term * i1_val + cheb_val / tl.maximum(x_f32, 1e-30)
    small_result = tl.exp(x_f32) * k1_small

    # ================================================================
    # Mid region: 2.0 < x <= 8.0
    # Polynomial fit for sqrt(x) * scaled_K1(x) in t = 1/x
    # scaled_K1(x) = poly(1/x) / sqrt(x)
    # ================================================================
    t_mid = 1.0 / ax
    # Horner evaluation of degree-9 polynomial fitted on [2, 5]
    poly_mid = 1.787118672412693
    poly_mid = 8.623190654362435 + t_mid * poly_mid
    poly_mid = -31.24007105377005 + t_mid * poly_mid
    poly_mid = 39.94627099526494 + t_mid * poly_mid
    poly_mid = -27.61325465617906 + t_mid * poly_mid
    poly_mid = 11.54410129937854 + t_mid * poly_mid
    poly_mid = -2.958011161687468 + t_mid * poly_mid
    poly_mid = 0.352116615179864 + t_mid * poly_mid
    poly_mid = 0.424654857840721 + t_mid * poly_mid
    poly_mid = 1.255087565900909 + t_mid * poly_mid
    mid_result = poly_mid / tl.sqrt(ax)

    # ================================================================
    # Large region: x > 8.0
    # Use asymptotic expansion for scaled_K1 directly:
    #   scaled_K1(x) = sqrt(pi/(2x)) * sum_{k=0}^5 (1,k) / (2x)^k
    # where (1,k) are the Bessel asymptotic coefficients.
    # ================================================================
    t_large = 1.0 / ax
    # Horner polynomial: 1 + 3t/8 - 15t^2/128 + 105t^3/1024 - 4725t^4/32768 + 72765t^5/262144
    poly_large = 1.0 + t_large * (
        0.375  # 3/8
        + t_large
        * (
            -0.1171875  # -15/128
            + t_large
            * (
                0.1025390625  # 105/1024
                + t_large
                * (
                    -0.144195556640625  # -4725/32768
                    + t_large * 0.277576446533203125  # 72765/262144
                )
            )
        )
    )
    # sqrt(pi/2) = 1.2533141373155001
    big_result = 1.2533141373155001 * poly_large / tl.sqrt(ax)

    # Combine results: three regions
    use_small = ax <= 2.0
    use_mid = (ax > 2.0) & (ax <= 8.0)
    result = tl.where(
        use_small, small_result, tl.where(use_mid, mid_result, big_result)
    )

    # Handle singularity at x=0 -> infinity
    result = tl.where(is_zero, float("inf"), result)
    # Handle negative inputs: return nan (matching PyTorch behavior)
    result = tl.where(is_negative, float("nan"), result)

    # Cast back to output dtype and store
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _k1_kernel_unmasked(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)
    x_f32 = x.to(tl.float32)

    # Handle non-positive values
    zero = 0.0
    ax = tl.abs(x_f32)
    is_zero = ax < 1e-30
    is_negative = x_f32 < zero

    # ================================================================
    # Small region: 0 < x <= 2.0
    # scaled_K1(x) = exp(x) * K1(x)
    # K1(x) = log(x/2) * I1(x) + P(x) / x
    # P(x) evaluated via Chebyshev series in w = x - 1 (degree 10, 11 terms).
    # ================================================================

    # --- Step 1: Compute I1(x) via Cephes polynomial (valid for x <= 3.75) ---
    y_i1 = x_f32 / 3.75
    y2_i1 = y_i1 * y_i1
    p_i1 = 0.00032411
    p_i1 = 0.00301532 + y2_i1 * p_i1
    p_i1 = 0.02658733 + y2_i1 * p_i1
    p_i1 = 0.15084934 + y2_i1 * p_i1
    p_i1 = 0.51498869 + y2_i1 * p_i1
    p_i1 = 0.87890594 + y2_i1 * p_i1
    p_i1 = 0.5 + y2_i1 * p_i1
    i1_val = x_f32 * p_i1

    # --- Step 2: Chebyshev series for P(x) at w = x - 1 ---
    # Clenshaw recurrence (3-state, correct update order):
    #   b_2 = b_1; b_1 = b_0; b_0 = c_k + 2*y*b_1 - b_2
    # Result: (b_0 - b_2) / 2
    y_cheb = x_f32 - 1.0
    b0 = 0.0
    b1 = 0.0

    # k=10: T_10 coefficient
    b2 = b1
    b1 = b0
    b0 = -3.662620162661560e-09 + 2.0 * y_cheb * b1 - b2

    # k=9: T_9 coefficient
    b2 = b1
    b1 = b0
    b0 = -5.350617072358099e-08 + 2.0 * y_cheb * b1 - b2

    # k=8: T_8 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.045185163440483e-06 + 2.0 * y_cheb * b1 - b2

    # k=7: T_7 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.201112421320492e-05 + 2.0 * y_cheb * b1 - b2

    # k=6: T_6 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.699042469646379e-04 + 2.0 * y_cheb * b1 - b2

    # k=5: T_5 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.426485944971723e-03 + 2.0 * y_cheb * b1 - b2

    # k=4: T_4 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.225928401715418e-02 + 2.0 * y_cheb * b1 - b2

    # k=3: T_3 coefficient
    b2 = b1
    b1 = b0
    b0 = -6.428271235627786e-02 + 2.0 * y_cheb * b1 - b2

    # k=2: T_2 coefficient
    b2 = b1
    b1 = b0
    b0 = -1.767199860308625e-01 + 2.0 * y_cheb * b1 - b2

    # k=1: T_1 coefficient
    b2 = b1
    b1 = b0
    b0 = -2.944128552518444e-01 + 2.0 * y_cheb * b1 - b2

    # k=0: T_0 coefficient (doubled for Clenshaw halving convention)
    b2 = b1
    b1 = b0
    b0 = 1.658032209918573e00 + 2.0 * y_cheb * b1 - b2

    # Final Chebyshev value: (b_0 - b_2) / 2
    cheb_val = (b0 - b2) / 2.0

    # --- Step 3: Compute K1(x) and scaled_K1(x) ---
    # K1(x) = log(x/2) * I1(x) + cheb_val / x
    # scaled_K1(x) = exp(x) * K1(x)
    log_term = tl.log(tl.maximum(x_f32, 1e-30) / 2.0)
    k1_small = log_term * i1_val + cheb_val / tl.maximum(x_f32, 1e-30)
    small_result = tl.exp(x_f32) * k1_small

    # ================================================================
    # Mid region: 2.0 < x <= 8.0
    # Polynomial fit for sqrt(x) * scaled_K1(x) in t = 1/x
    # scaled_K1(x) = poly(1/x) / sqrt(x)
    # ================================================================
    t_mid = 1.0 / ax
    # Horner evaluation of degree-9 polynomial fitted on [2, 5]
    poly_mid = 1.787118672412693
    poly_mid = 8.623190654362435 + t_mid * poly_mid
    poly_mid = -31.24007105377005 + t_mid * poly_mid
    poly_mid = 39.94627099526494 + t_mid * poly_mid
    poly_mid = -27.61325465617906 + t_mid * poly_mid
    poly_mid = 11.54410129937854 + t_mid * poly_mid
    poly_mid = -2.958011161687468 + t_mid * poly_mid
    poly_mid = 0.352116615179864 + t_mid * poly_mid
    poly_mid = 0.424654857840721 + t_mid * poly_mid
    poly_mid = 1.255087565900909 + t_mid * poly_mid
    mid_result = poly_mid / tl.sqrt(ax)

    # ================================================================
    # Large region: x > 8.0
    # Use asymptotic expansion for scaled_K1 directly:
    #   scaled_K1(x) = sqrt(pi/(2x)) * sum_{k=0}^5 (1,k) / (2x)^k
    # where (1,k) are the Bessel asymptotic coefficients.
    # ================================================================
    t_large = 1.0 / ax
    # Horner polynomial: 1 + 3t/8 - 15t^2/128 + 105t^3/1024 - 4725t^4/32768 + 72765t^5/262144
    poly_large = 1.0 + t_large * (
        0.375  # 3/8
        + t_large
        * (
            -0.1171875  # -15/128
            + t_large
            * (
                0.1025390625  # 105/1024
                + t_large
                * (
                    -0.144195556640625  # -4725/32768
                    + t_large * 0.277576446533203125  # 72765/262144
                )
            )
        )
    )
    # sqrt(pi/2) = 1.2533141373155001
    big_result = 1.2533141373155001 * poly_large / tl.sqrt(ax)

    # Combine results: three regions
    use_small = ax <= 2.0
    use_mid = (ax > 2.0) & (ax <= 8.0)
    result = tl.where(
        use_small, small_result, tl.where(use_mid, mid_result, big_result)
    )

    # Handle singularity at x=0 -> infinity
    result = tl.where(is_zero, float("inf"), result)
    # Handle negative inputs: return nan (matching PyTorch behavior)
    result = tl.where(is_negative, float("nan"), result)

    # Cast back to output dtype and store
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty))


def _launch_k1(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    x_contig = x.contiguous() if not x.is_contiguous() else x
    out_contig = out.contiguous() if not out.is_contiguous() else out
    block_size, num_warps, masked = _pick_block(n_elements)
    if masked:
        grid = (triton.cdiv(n_elements, block_size),)
        _k1_kernel[grid](
            x_contig,
            out_contig,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            isCloseVectorization=IS_CLOSE_VECTORIZATION,
        )
    else:
        grid = (n_elements // block_size,)
        _k1_kernel_unmasked[grid](
            x_contig,
            out_contig,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            unroll_num=UNROLL_NUM,
            buffer_size_limit=BUFFER_SIZE_LIMIT,
            isCloseMemoryAsync=IS_CLOSE_MEMORY_ASYNC,
            isCloseVectorization=IS_CLOSE_VECTORIZATION,
        )
    if out_contig.data_ptr() != out.data_ptr():
        out.copy_(out_contig)
    return out


def special_scaled_modified_bessel_k1(x: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K1")
    if not x.is_cuda:
        raise ValueError(
            "special_scaled_modified_bessel_k1: input tensor must be on CUDA device"
        )
    # Result dtype follows PyTorch's floating type behavior
    out_dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    out = torch.empty_like(x, dtype=out_dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _launch_k1(x, out)
    return out


def special_scaled_modified_bessel_k1_out(
    x: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SCALED_MODIFIED_BESSEL_K1_OUT")
    if not (x.is_cuda and out.is_cuda):
        raise ValueError(
            "special_scaled_modified_bessel_k1_out: input and output tensors must be on CUDA device"
        )
    if not out.is_floating_point():
        raise TypeError(
            "special_scaled_modified_bessel_k1_out: output tensor must be a floating point type"
        )
    if x.numel() != out.numel():
        raise ValueError(
            "special_scaled_modified_bessel_k1_out: input and output must have the same number of elements"
        )
    with torch_device_fn.device(x.device):
        _launch_k1(x, out)
    return out
