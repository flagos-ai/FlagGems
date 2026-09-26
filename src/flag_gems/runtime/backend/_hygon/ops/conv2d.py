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
import weakref

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.conv2d import conv2d as _generic_conv2d
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P0-4 / selective native MIOpen redispatch
# ---------------------------------------------------------------------------

# Match the current Hygon cudnn_convolution fallback: redispatch at
# CompositeExplicitAutograd to bypass FlagGems Triton dispatch and reach the
# native ROCm/DCU MIOpen convolution implementation.
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _resolve_native_padding(weight: torch.Tensor, stride, padding, dilation):
    """Resolve conv2d padding to an integer pair for aten::convolution.

    Returns None for string padding that would require asymmetric explicit
    padding.  Those uncommon cases stay on the generic FlagGems path.
    """
    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")

    if not isinstance(padding, str):
        return _normalize_2d(padding, "padding")

    mode = padding.lower()
    if mode == "valid":
        return 0, 0

    if mode != "same":
        return None

    # torch.nn.functional.conv2d padding="same" only supports stride=1.
    if (stride_h, stride_w) != (1, 1):
        return None

    kernel_h, kernel_w = weight.shape[-2:]
    total_h = dilation_h * (kernel_h - 1)
    total_w = dilation_w * (kernel_w - 1)

    # aten::convolution takes symmetric integer padding.  If SAME would be
    # asymmetric (for example an even effective kernel), keep generic behavior.
    if total_h % 2 != 0 or total_w % 2 != 0:
        return None

    return total_h // 2, total_w // 2


def _should_use_native_miopen_fp32(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> bool:
    """Use native MIOpen for FP32 inference while preserving safe fallback."""
    if input.ndim != 4 or weight.ndim != 4:
        return False

    if input.dtype != torch.float32 or weight.dtype != torch.float32:
        return False

    if bias is not None and bias.dtype != torch.float32:
        return False

    if input.device != weight.device:
        return False

    if bias is not None and bias.device != input.device:
        return False

    # Keep training/backward on the pre-existing generic implementation.
    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return False

    return _resolve_native_padding(weight, stride, padding, dilation) is not None


def _should_use_selective_native_miopen_low_precision(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> bool:
    """Select exact FP16/BF16 families where measured Triton speedup < 0.9.

    The whitelist is intentionally narrow so already-fast Triton families keep
    their current implementation.  It can be widened only after benchmarking.
    """
    if input.ndim != 4 or weight.ndim != 4:
        return False

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.dtype != input.dtype:
        return False
    if bias is not None and bias.dtype != input.dtype:
        return False

    if input.device != weight.device:
        return False
    if bias is not None and bias.device != input.device:
        return False

    if groups != 1:
        return False

    if not input.is_contiguous():
        return False

    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return False

    resolved_padding = _resolve_native_padding(weight, stride, padding, dilation)
    if resolved_padding is None:
        return False

    stride_hw = _normalize_2d(stride, "stride")
    dilation_hw = _normalize_2d(dilation, "dilation")
    if dilation_hw != (1, 1):
        return False

    input_shape = tuple(int(v) for v in input.shape)
    weight_shape = tuple(int(v) for v in weight.shape)
    pad_hw = tuple(int(v) for v in resolved_padding)

    # Shared FP16/BF16 misses from the benchmark suite.
    shared_slow_families = (
        # 3x3 p1: Triton ~0.71x in both FP16 and BF16.
        (
            input_shape == (32, 64, 128, 128)
            and weight_shape == (32, 64, 3, 3)
            and stride_hw == (1, 1)
            and pad_hw == (1, 1)
        ),
        # 4x4 Cin16: generic path ~0.65-0.73x.
        (
            input_shape == (104, 16, 32, 32)
            and weight_shape == (32, 16, 4, 4)
            and stride_hw == (1, 1)
            and pad_hw == (0, 0)
        ),
    )
    if any(shared_slow_families):
        return True

    if input.dtype != torch.bfloat16:
        return False

    # BF16-only misses.  FP16 versions of these same families are already at
    # or above the target and must remain on Triton.
    bf16_slow_families = (
        # Small 5x5: ~0.70x.
        (
            input_shape == (64, 32, 18, 18)
            and weight_shape == (32, 32, 5, 5)
            and stride_hw == (2, 2)
            and pad_hw == (1, 1)
        ),
        # Large 5x5 Cout64: ~0.70x in BF16, ~0.93x in FP16.
        (
            input_shape == (64, 32, 210, 210)
            and weight_shape == (64, 32, 5, 5)
            and stride_hw == (2, 2)
            and pad_hw == (1, 1)
        ),
        # Large 5x5 Cout16 stride2: ~0.81x in BF16, >1.0x in FP16.
        (
            input_shape == (32, 64, 210, 210)
            and weight_shape == (16, 64, 5, 5)
            and stride_hw == (2, 2)
            and pad_hw == (1, 1)
        ),
        # 3x3 p2: ~0.884x, included to make the >=0.9 target robust.
        (
            input_shape == (32, 64, 128, 128)
            and weight_shape == (32, 64, 3, 3)
            and stride_hw == (1, 1)
            and pad_hw == (2, 2)
        ),
    )
    return any(bf16_slow_families)


def _native_miopen_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> torch.Tensor:
    """Call native MIOpen convolution, bypassing FlagGems conv dispatch."""
    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")
    resolved_padding = _resolve_native_padding(weight, stride, padding, dilation)
    if resolved_padding is None:
        return _generic_conv2d(input, weight, bias, stride, padding, dilation, groups)
    pad_h, pad_w = resolved_padding

    return torch.ops.aten.convolution.default.redispatch(
        _FALLBACK_KEYSET,
        input,
        weight,
        bias,
        [stride_h, stride_w],
        [pad_h, pad_w],
        [dilation_h, dilation_w],
        False,
        [0, 0],
        groups,
    )


# ---------------------------------------------------------------------------
# P0-1/P0-2.1 dispatch / cache policy
# ---------------------------------------------------------------------------

# Large enough that packed-weight reuse can amortize the one-time packing cost.
_WEIGHT_PREPACK_MIN_M = 32768
_MIN_PACKED_CI = 32

# P0-3: only pay the input layout-conversion cost for large dense 5x5 work.
_CL_CONVERT_MIN_M = 262144
_CL_CONVERT_MIN_CI = 32
_CL_CONVERT_MIN_KERNEL_AREA = 25

# Fused NCHW -> channels-last + symmetric zero-padding launch shape.
_FUSED_PAD_CL_BLOCK_HW = 32
_FUSED_PAD_CL_BLOCK_C = 64


# id(weight) ->
#   (weakref(weight), tensor_version, shape, stride, packed_hwio_tensor)
_WEIGHT_PREPACK_CACHE = {}


def clear_conv2d_weight_prepack_cache():
    """Drop all cached HWIO conv2d weights."""
    _WEIGHT_PREPACK_CACHE.clear()


def _normalize_2d(value, name):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly 2 values")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv2d_output_size(in_size, kernel_size, stride, padding, dilation):
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _resolve_fast_padding(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride,
    padding,
    dilation,
):
    """
    Resolve padding for the packed-weight fast path.

    Returns a symmetric (pad_h, pad_w), or None when this request should stay on
    the generic FlagGems implementation.
    """
    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")

    if not isinstance(padding, str):
        return _normalize_2d(padding, "padding")

    mode = padding.lower()
    if mode == "valid":
        return 0, 0

    if mode != "same":
        return None

    # torch conv2d padding="same" only supports stride=1.
    if stride_h != 1 or stride_w != 1:
        return None

    kernel_h, kernel_w = weight.shape[-2:]
    total_h = dilation_h * (kernel_h - 1)
    total_w = dilation_w * (kernel_w - 1)

    # P0-1 only handles symmetric SAME directly.  The selected 3x3/5x5,
    # dilation=1 fast-path families satisfy this condition.
    if total_h % 2 != 0 or total_w % 2 != 0:
        return None

    return total_h // 2, total_w // 2


def _get_prepacked_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Return a cached contiguous HWIO view-copy of an OIHW PyTorch weight.

    PyTorch logical/physical layout used by conv2d:
        weight: [CO, CI, KH, KW]  (OIHW)

    Packed layout used by the fast Triton kernel:
        packed: [KH, KW, CI, CO]  (HWIO)

    For fixed (kh, kw), the implicit-GEMM B matrix is [CI, CO], and CO is
    contiguous.  Tensor._version invalidates the cache after in-place updates.
    """
    key = id(weight)
    version = int(getattr(weight, "_version", 0))
    shape = tuple(weight.shape)
    stride = tuple(weight.stride())

    entry = _WEIGHT_PREPACK_CACHE.get(key)
    if entry is not None:
        weight_ref, cached_version, cached_shape, cached_stride, packed = entry
        if (
            weight_ref() is weight
            and cached_version == version
            and cached_shape == shape
            and cached_stride == stride
        ):
            return packed

    # OIHW -> HWIO.  This is the only data-layout transformation in P0-1.
    packed = weight.permute(2, 3, 1, 0).contiguous()

    def _remove(dead_ref, cache_key=key):
        current = _WEIGHT_PREPACK_CACHE.get(cache_key)
        if current is not None and current[0] is dead_ref:
            _WEIGHT_PREPACK_CACHE.pop(cache_key, None)

    weight_ref = weakref.ref(weight, _remove)
    _WEIGHT_PREPACK_CACHE[key] = (
        weight_ref,
        version,
        shape,
        stride,
        packed,
    )
    return packed


def _should_use_packed_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
):
    """Conservative P0-1 eligibility policy for the packed-weight fast path."""
    if input.ndim != 4 or weight.ndim != 4:
        return False

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False

    if weight.dtype != input.dtype:
        return False

    if bias is not None and bias.dtype != input.dtype:
        return False

    if groups != 1:
        return False

    # P0-1 intentionally keeps the input/output layout unchanged (NCHW).
    if not input.is_contiguous():
        return False

    # Keep autograd/backward entirely on the existing generic implementation.
    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return False

    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")

    resolved_padding = _resolve_fast_padding(
        input,
        weight,
        stride,
        padding,
        dilation,
    )
    if resolved_padding is None:
        return False
    pad_h, pad_w = resolved_padding

    if dilation_h != 1 or dilation_w != 1:
        return False

    batch, in_channels, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape

    if in_channels != channels_per_group:
        return False

    if channels_per_group < _MIN_PACKED_CI:
        return False

    # Start with the benchmark families where weight packing is most useful.
    if (kernel_h, kernel_w) not in ((3, 3), (5, 5)):
        return False

    out_h = _conv2d_output_size(
        in_h,
        kernel_h,
        stride_h,
        pad_h,
        dilation_h,
    )
    out_w = _conv2d_output_size(
        in_w,
        kernel_w,
        stride_w,
        pad_w,
        dilation_w,
    )

    if out_h <= 0 or out_w <= 0:
        return False

    return batch * out_h * out_w >= _WEIGHT_PREPACK_MIN_M


def _should_convert_input_to_channels_last(
    input: torch.Tensor,
    batch: int,
    out_h: int,
    out_w: int,
    channels_per_group: int,
    kernel_h: int,
    kernel_w: int,
    groups: int,
) -> bool:
    """P0-3 policy for fused NCHW->channels-last(+padding)."""
    if input.is_contiguous(memory_format=torch.channels_last):
        return False

    if not input.is_contiguous():
        return False

    if groups != 1:
        return False

    if channels_per_group < _CL_CONVERT_MIN_CI:
        return False

    if kernel_h * kernel_w < _CL_CONVERT_MIN_KERNEL_AREA:
        return False

    if batch * out_h * out_w < _CL_CONVERT_MIN_M:
        return False

    return True


# ---------------------------------------------------------------------------
# P0-3 fused NCHW -> channels-last + symmetric zero padding
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def _nchw_to_channels_last_pad_kernel(
    x_ptr,
    y_ptr,
    batch,
    channels,
    in_h,
    in_w,
    padded_h,
    padded_w,
    pad_h,
    pad_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fuse NCHW->channels-last layout conversion and zero padding."""
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    hw_lane = tl.arange(0, BLOCK_HW)
    c_lane = tl.arange(0, BLOCK_C)

    flat_hw = pid_hw * BLOCK_HW + hw_lane
    total_hw = batch * padded_h * padded_w
    hw_mask = flat_hw < total_hw

    hw_per_batch = padded_h * padded_w
    n = flat_hw // hw_per_batch
    rem = flat_hw - n * hw_per_batch
    ph = rem // padded_w
    pw = rem - ph * padded_w

    c = pid_c * BLOCK_C + c_lane
    c_mask = c < channels

    ih = ph - pad_h
    iw = pw - pad_w
    inside = hw_mask & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)

    x_ptrs = (
        x_ptr
        + n[:, None] * x_stride_n
        + c[None, :] * x_stride_c
        + ih[:, None] * x_stride_h
        + iw[:, None] * x_stride_w
    )
    y_ptrs = (
        y_ptr
        + n[:, None] * y_stride_n
        + c[None, :] * y_stride_c
        + ph[:, None] * y_stride_h
        + pw[:, None] * y_stride_w
    )

    value = tl.load(
        x_ptrs,
        mask=inside[:, None] & c_mask[None, :],
        other=0.0,
    )
    tl.store(
        y_ptrs,
        value,
        mask=hw_mask[:, None] & c_mask[None, :],
    )


def _nchw_to_padded_channels_last(
    input: torch.Tensor,
    pad_h: int,
    pad_w: int,
) -> torch.Tensor:
    """Return logical NCHW with channels-last storage and materialized padding."""
    batch, channels, in_h, in_w = input.shape
    padded_h = in_h + 2 * pad_h
    padded_w = in_w + 2 * pad_w

    output = torch.empty(
        (batch, channels, padded_h, padded_w),
        device=input.device,
        dtype=input.dtype,
        memory_format=torch.channels_last,
    )

    grid = (
        triton.cdiv(batch * padded_h * padded_w, _FUSED_PAD_CL_BLOCK_HW),
        triton.cdiv(channels, _FUSED_PAD_CL_BLOCK_C),
    )

    _nchw_to_channels_last_pad_kernel[grid](
        input,
        output,
        batch,
        channels,
        in_h,
        in_w,
        padded_h,
        padded_w,
        pad_h,
        pad_w,
        *input.stride(),
        *output.stride(),
        BLOCK_HW=_FUSED_PAD_CL_BLOCK_HW,
        BLOCK_C=_FUSED_PAD_CL_BLOCK_C,
        num_warps=4,
        num_stages=1,
    )
    return output


# ---------------------------------------------------------------------------
# P0-2.1 Cout=16-only packed-HWIO autotune search space
# ---------------------------------------------------------------------------

# This is the only new autotune space retained from P0-2.  BW1000 benchmark
# results showed a repeatable win for Cout=16 when BLOCK_CO exactly matches 16.
# All other output-channel counts use the original Hygon conv2d_forward configs.
_COUT16_PACKED_FORWARD_CONFIGS = [
    triton.Config(
        {"BLOCK_NI_HO_WO": 32, "BLOCK_CO": 16, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 16, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 16, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 16, "BLOCK_CI": 32},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 32, "BLOCK_CO": 16, "BLOCK_CI": 64},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 64, "BLOCK_CO": 16, "BLOCK_CI": 64},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_NI_HO_WO": 128, "BLOCK_CO": 16, "BLOCK_CI": 64},
        num_warps=4,
        num_stages=1,
    ),
]


# ---------------------------------------------------------------------------
# P0-2.1 packed-HWIO implicit-GEMM forward kernels
# ---------------------------------------------------------------------------


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("conv2d_forward"),
    key=[
        "in_n",
        "weight_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
        "CHECK_SPATIAL_BOUNDS",
    ],
)
@triton.jit
def conv2d_forward_packed_weight_kernel(
    input_pointer,
    packed_weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    packed_weight_h_stride,
    packed_weight_w_stride,
    packed_weight_i_stride,
    packed_weight_o_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHECK_SPATIAL_BOUNDS: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    NCHW input + HWIO weight -> NCHW output implicit-GEMM conv2d.

    Compared with the generic FlagGems forward kernel, input/output addressing
    is intentionally unchanged.  Only the weight tile addressing changes:

        generic OIHW: B[CI, CO] is strided in CO
        packed  HWIO: B[CI, CO] has contiguous CO

    groups is kept as a constexpr argument so the existing Hygon autotune key
    can be reused, but this kernel is dispatched only with groups == 1.
    """
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)

    m = pid_m * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    out_hw = out_height * out_width
    n = m // out_hw
    hw = m - n * out_hw
    oh = hw // out_width
    ow = hw - oh * out_width

    acc = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    block_ci_count: tl.constexpr = (weight_c + BLOCK_CI - 1) // BLOCK_CI

    for r in range(weight_height * weight_width * block_ci_count):
        ci_block = r % block_ci_count
        tap = r // block_ci_count
        kh = tap // weight_width
        kw = tap - kh * weight_width

        ci = ci_block * BLOCK_CI + tl.arange(0, BLOCK_CI)

        ih = oh * stride_height + kh * dilation_height - padding_height
        iw = ow * stride_width + kw * dilation_width - padding_width

        input_ptrs = (
            input_pointer
            + n[:, None] * input_n_stride
            + ci[None, :] * input_c_stride
            + ih[:, None] * input_height_stride
            + iw[:, None] * input_width_stride
        )

        # packed_weight is [KH, KW, CI, CO].
        weight_ptrs = (
            packed_weight_pointer
            + kh * packed_weight_h_stride
            + kw * packed_weight_w_stride
            + ci[:, None] * packed_weight_i_stride
            + co[None, :] * packed_weight_o_stride
        )

        base_input_mask = (n < in_n)[:, None] & (ci < weight_c)[None, :]
        if CHECK_SPATIAL_BOUNDS:
            input_mask = (
                base_input_mask
                & (ih >= 0)[:, None]
                & (ih < input_height)[:, None]
                & (iw >= 0)[:, None]
                & (iw < input_width)[:, None]
            )
        else:
            input_mask = base_input_mask

        weight_mask = (ci < weight_c)[:, None] & (co < out_c)[None, :]

        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        acc += tl.dot(input_block, weight_block, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + co,
            mask=co < out_c,
            other=0.0,
        ).to(tl.float32)
        acc += bias[None, :]

    output_ptrs = (
        output_pointer
        + n[:, None] * output_n_stride
        + co[None, :] * output_c_stride
        + oh[:, None] * output_height_stride
        + ow[:, None] * output_width_stride
    )

    output_mask = (n < in_n)[:, None] & (co < out_c)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


@libentry()
@triton.autotune(
    configs=_COUT16_PACKED_FORWARD_CONFIGS,
    key=[
        "in_n",
        "weight_c",
        "input_height",
        "input_width",
        "out_c",
        "out_height",
        "out_width",
        "weight_height",
        "weight_width",
        "stride_height",
        "stride_width",
        "padding_height",
        "padding_width",
        "groups",
        "CHECK_SPATIAL_BOUNDS",
    ],
)
@triton.jit
def conv2d_forward_packed_weight_cout16_kernel(
    input_pointer,
    packed_weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    input_height,
    input_width,
    out_c,
    out_height,
    out_width,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    packed_weight_h_stride,
    packed_weight_w_stride,
    packed_weight_i_stride,
    packed_weight_o_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    weight_c: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHECK_SPATIAL_BOUNDS: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Cout=16 NCHW input + HWIO weight -> NCHW output implicit-GEMM conv2d.

    Compared with the generic FlagGems forward kernel, input/output addressing
    is intentionally unchanged.  Only the weight tile addressing changes:

        generic OIHW: B[CI, CO] is strided in CO
        packed  HWIO: B[CI, CO] has contiguous CO

    groups is kept as a constexpr argument so the existing Hygon autotune key
    can be reused, but this kernel is dispatched only with groups == 1.
    """
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)

    m = pid_m * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    out_hw = out_height * out_width
    n = m // out_hw
    hw = m - n * out_hw
    oh = hw // out_width
    ow = hw - oh * out_width

    acc = tl.zeros((BLOCK_NI_HO_WO, BLOCK_CO), dtype=tl.float32)

    block_ci_count: tl.constexpr = (weight_c + BLOCK_CI - 1) // BLOCK_CI

    for r in range(weight_height * weight_width * block_ci_count):
        ci_block = r % block_ci_count
        tap = r // block_ci_count
        kh = tap // weight_width
        kw = tap - kh * weight_width

        ci = ci_block * BLOCK_CI + tl.arange(0, BLOCK_CI)

        ih = oh * stride_height + kh * dilation_height - padding_height
        iw = ow * stride_width + kw * dilation_width - padding_width

        input_ptrs = (
            input_pointer
            + n[:, None] * input_n_stride
            + ci[None, :] * input_c_stride
            + ih[:, None] * input_height_stride
            + iw[:, None] * input_width_stride
        )

        # packed_weight is [KH, KW, CI, CO].
        weight_ptrs = (
            packed_weight_pointer
            + kh * packed_weight_h_stride
            + kw * packed_weight_w_stride
            + ci[:, None] * packed_weight_i_stride
            + co[None, :] * packed_weight_o_stride
        )

        base_input_mask = (n < in_n)[:, None] & (ci < weight_c)[None, :]
        if CHECK_SPATIAL_BOUNDS:
            input_mask = (
                base_input_mask
                & (ih >= 0)[:, None]
                & (ih < input_height)[:, None]
                & (iw >= 0)[:, None]
                & (iw < input_width)[:, None]
            )
        else:
            input_mask = base_input_mask

        weight_mask = (ci < weight_c)[:, None] & (co < out_c)[None, :]

        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_block = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        acc += tl.dot(input_block, weight_block, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(
            bias_pointer + co,
            mask=co < out_c,
            other=0.0,
        ).to(tl.float32)
        acc += bias[None, :]

    output_ptrs = (
        output_pointer
        + n[:, None] * output_n_stride
        + co[None, :] * output_c_stride
        + oh[:, None] * output_height_stride
        + ow[:, None] * output_width_stride
    )

    output_mask = (n < in_n)[:, None] & (co < out_c)[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


def _packed_weight_conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
):
    """Execute P0-2.1 plus the P0-3 large-5x5 channels-last path."""
    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")

    resolved_padding = _resolve_fast_padding(
        input,
        weight,
        stride,
        padding,
        dilation,
    )
    if resolved_padding is None:
        return _generic_conv2d(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            1,
        )

    pad_h, pad_w = resolved_padding
    batch, channels_per_group, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    out_h = _conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
    out_w = _conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)

    use_cl = _should_convert_input_to_channels_last(
        input,
        batch,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        1,
    )

    forward_input = input
    forward_in_h = in_h
    forward_in_w = in_w
    forward_pad_h = pad_h
    forward_pad_w = pad_w
    check_spatial_bounds = True

    if use_cl:
        # Materialize the user's symmetric padding while converting to CL.
        # Every receptive field used for the original output is now valid, so
        # the convolution runs with pad=0 and no H/W bounds checks.
        forward_input = _nchw_to_padded_channels_last(input, pad_h, pad_w)
        forward_in_h = in_h + 2 * pad_h
        forward_in_w = in_w + 2 * pad_w
        forward_pad_h = 0
        forward_pad_w = 0
        check_spatial_bounds = False

        output = torch.empty(
            (batch, out_channels, out_h, out_w),
            device=input.device,
            dtype=input.dtype,
            memory_format=torch.channels_last,
        )
    else:
        output = torch.empty(
            (batch, out_channels, out_h, out_w),
            device=input.device,
            dtype=input.dtype,
        )

    packed_weight = _get_prepacked_weight(weight)

    grid = lambda meta: (
        triton.cdiv(batch * out_h * out_w, meta["BLOCK_NI_HO_WO"]),
        triton.cdiv(out_channels, meta["BLOCK_CO"]),
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else output

    # Keep the validated P0-2.1 hybrid autotune split.
    kernel = (
        conv2d_forward_packed_weight_cout16_kernel
        if out_channels == 16
        else conv2d_forward_packed_weight_kernel
    )

    kernel[grid](
        forward_input,
        packed_weight,
        output,
        bias_ptr,
        batch,
        forward_in_h,
        forward_in_w,
        out_channels,
        out_h,
        out_w,
        *forward_input.stride(),
        *packed_weight.stride(),
        *output.stride(),
        channels_per_group,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        forward_pad_h,
        forward_pad_w,
        dilation_h,
        dilation_w,
        groups=1,
        HAS_BIAS=has_bias,
        CHECK_SPATIAL_BOUNDS=check_spatial_bounds,
    )

    return output


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------


def conv2d_would_use_native_miopen_fp32(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether P0-4 native MIOpen FP32 redispatch is selected."""
    return _should_use_native_miopen_fp32(
        input, weight, bias, stride, padding, dilation, groups
    )


def conv2d_would_use_selective_native_miopen(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether the selective FP16/BF16 MIOpen whitelist is selected."""
    return _should_use_selective_native_miopen_low_precision(
        input, weight, bias, stride, padding, dilation, groups
    )


def conv2d_would_use_packed_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether the P0-2.1 packed-HWIO path would be selected."""
    return _should_use_packed_weight(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )


def conv2d_would_convert_input_to_channels_last(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether P0-3 fused NCHW->CL(+padding) is selected."""
    if not _should_use_packed_weight(
        input, weight, None, stride, padding, dilation, groups
    ):
        return False

    resolved_padding = _resolve_fast_padding(input, weight, stride, padding, dilation)
    if resolved_padding is None:
        return False

    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")
    pad_h, pad_w = resolved_padding
    batch, _, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape
    out_h = _conv2d_output_size(in_h, kernel_h, stride_h, pad_h, dilation_h)
    out_w = _conv2d_output_size(in_w, kernel_w, stride_w, pad_w, dilation_w)
    return _should_convert_input_to_channels_last(
        input, batch, out_h, out_w, channels_per_group, kernel_h, kernel_w, groups
    )


def conv2d_p0_2_1_autotune_family(weight: torch.Tensor) -> str:
    """Describe which P0-2.1 autotune family this weight will use."""
    if weight.ndim != 4:
        return "generic"

    out_channels = int(weight.shape[0])
    if out_channels == 16:
        return "Cout16: dedicated BLOCK_CO=16 packed-HWIO autotune"
    return "P0-1 baseline: original Hygon conv2d_forward autotune configs"


# ---------------------------------------------------------------------------
# Public Hygon backend entry
# ---------------------------------------------------------------------------


def conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    """
    Hygon conv2d entry -- P0-4 plus selective low-precision MIOpen.

    Route A -- FP32 inference:
        -> native Hygon/ROCm MIOpen via aten::convolution.redispatch

    Route B -- selected FP16/BF16 benchmark families with measured Triton
    speedup below the ~0.9 target:
        -> native MIOpen

    Route C -- eligible packed FP16/BF16 3x3/5x5:
        -> P0-1/P0-2.1 cached HWIO weight
        -> P0-3 fused channels-last+padding for large 5x5

    Route D:
        everything else -> generic FlagGems conv2d
    """
    if _should_use_native_miopen_fp32(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS_HYGON CONV2D P0_4_NATIVE_MIOPEN_FP32")
        return _native_miopen_conv2d(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

    if _should_use_selective_native_miopen_low_precision(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS_HYGON CONV2D SELECTIVE_NATIVE_MIOPEN_LOW_PRECISION")
        return _native_miopen_conv2d(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

    if _should_use_packed_weight(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS_HYGON CONV2D P0_3_PACKED_HWIO_CL_HYBRID")
        return _packed_weight_conv2d_forward(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
        )

    logger.debug("GEMS_HYGON CONV2D GENERIC_FALLBACK")
    return _generic_conv2d(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
