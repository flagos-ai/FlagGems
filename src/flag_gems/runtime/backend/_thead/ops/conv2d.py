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

from flag_gems.ops.conv2d import conv2d as _generic_conv2d
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch policy
# ---------------------------------------------------------------------------

# Large enough for packed-weight reuse to amortize prepacking/cache lookup.
_WEIGHT_PREPACK_MIN_M = 32768
_MIN_PACKED_CI = 32

# Conservative V2 channels-last conversion policy, matching the shape family
# proven useful in the MThreads backend:
#   input=[64,32,210,210], weight=[64,32,5,5], stride=2, padding=1
_CL_CONVERT_MIN_M = 262144
_CL_CONVERT_MIN_CI = 32
_CL_CONVERT_MIN_KERNEL_AREA = 25

# Fused NCHW -> channels-last + zero-padding tile.
_FUSED_PAD_CL_BLOCK_HW = 32
_FUSED_PAD_CL_BLOCK_C = 64

# V3 fixed 3x3 path.  Keep it intentionally exact for the first experiment.
_SPECIAL_3X3_BLOCK_M = 128
_SPECIAL_3X3_BLOCK_K = 64
_SPECIAL_3X3_BLOCK_N = 32

# V4 tail / valid tiles.
_SPECIAL_3X3_TAIL_M = 16
_SPECIAL_3X3_VALID_M = 64

# id(weight) ->
#   (weakref(weight), tensor_version, shape, stride, packed_hwio_tensor)
_WEIGHT_PREPACK_CACHE = {}


def clear_conv2d_weight_prepack_cache():
    """Drop cached HWIO weights."""
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
    Resolve padding for the optimized forward path.

    Returns symmetric (pad_h, pad_w), or None when this request should stay on
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

    # torch.nn.functional.conv2d padding="same" only supports stride=1.
    if stride_h != 1 or stride_w != 1:
        return None

    kernel_h, kernel_w = weight.shape[-2:]
    total_h = dilation_h * (kernel_h - 1)
    total_w = dilation_w * (kernel_w - 1)

    # This optimized route handles symmetric SAME directly.
    # 3x3/5x5 with dilation=1 satisfy this.
    if total_h % 2 != 0 or total_w % 2 != 0:
        return None

    return total_h // 2, total_w // 2


def _get_prepacked_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Return cached contiguous HWIO weight.

    PyTorch weight:
        [CO, CI, KH, KW]  (OIHW)

    Packed weight:
        [KH, KW, CI, CO]  (HWIO)

    For each fixed (kh, kw), the implicit-GEMM B matrix is [CI, CO] with CO
    contiguous. Tensor._version invalidates the cache after in-place updates.
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
    """V1 packed-weight policy, kept intact for the V2 experiment."""
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

    # V1/V2 starts from standard contiguous NCHW inputs.
    if not input.is_contiguous():
        return False

    # Keep training/backward on the generic implementation for now.
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

    # Focus only on the benchmark families currently worth optimizing.
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
    """
    V2 policy: pay one NCHW->CL copy only for large dense 5x5-style workloads.

    The optimized entry already restricts dtype to FP16/BF16.  This helper is
    intentionally shape-aware and excludes the 128x128 3x3 family.
    """
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


def _should_use_special_3x3(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> bool:
    """
    V3 exact-shape fast path.

    First experiment intentionally targets only the benchmark family:
        N x 64 x 128 x 128
        32 x 64 x 3 x 3
        stride=1, padding=1, dilation=1, groups=1
        FP16/BF16

    Restricting the first version avoids regressions on p2/valid and makes the
    effect of the fixed nine-tap kernel easy to measure.
    """
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

    if not input.is_contiguous():
        return False

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

    if (stride_h, stride_w) != (1, 1):
        return False

    if (dilation_h, dilation_w) != (1, 1):
        return False

    if (pad_h, pad_w) != (1, 1):
        return False

    batch, in_channels, in_h, in_w = input.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weight.shape

    if in_channels != 64 or channels_per_group != 64:
        return False

    if out_channels != 32:
        return False

    if (kernel_h, kernel_w) != (3, 3):
        return False

    if (in_h, in_w) != (128, 128):
        return False

    out_h = _conv2d_output_size(in_h, 3, 1, 1, 1)
    out_w = _conv2d_output_size(in_w, 3, 1, 1, 1)

    # One program computes one complete M=128 output row.
    return out_h == 128 and out_w == _SPECIAL_3X3_BLOCK_M


def _special_3x3_common_eligible(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    dilation,
    groups,
) -> bool:
    """Common eligibility for the V3/V4 Cin64/Cout32 3x3 family."""
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

    if not input.is_contiguous():
        return False

    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        return False

    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")

    if (stride_h, stride_w) != (1, 1):
        return False

    if (dilation_h, dilation_w) != (1, 1):
        return False

    batch, in_channels, in_h, in_w = input.shape
    out_channels, channels_per_group, kernel_h, kernel_w = weight.shape

    if in_channels != 64 or channels_per_group != 64:
        return False

    if out_channels != 32:
        return False

    if (kernel_h, kernel_w) != (3, 3):
        return False

    if (in_h, in_w) != (128, 128):
        return False

    return True


def _should_use_special_3x3_p2(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> bool:
    """V4 p2 route: 128x128 input -> 130x130 output."""
    if not _special_3x3_common_eligible(
        input,
        weight,
        bias,
        stride,
        dilation,
        groups,
    ):
        return False

    resolved_padding = _resolve_fast_padding(
        input,
        weight,
        stride,
        padding,
        dilation,
    )
    if resolved_padding != (2, 2):
        return False

    out_h = _conv2d_output_size(128, 3, 1, 2, 1)
    out_w = _conv2d_output_size(128, 3, 1, 2, 1)
    return out_h == 130 and out_w == 130


def _should_use_special_3x3_valid(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
    groups,
) -> bool:
    """V4 valid route: 128x128 input -> 126x126 output."""
    if not _special_3x3_common_eligible(
        input,
        weight,
        bias,
        stride,
        dilation,
        groups,
    ):
        return False

    resolved_padding = _resolve_fast_padding(
        input,
        weight,
        stride,
        padding,
        dilation,
    )
    if resolved_padding != (0, 0):
        return False

    out_h = _conv2d_output_size(128, 3, 1, 0, 1)
    out_w = _conv2d_output_size(128, 3, 1, 0, 1)
    return out_h == 126 and out_w == 126


# ---------------------------------------------------------------------------
# Fused NCHW -> channels-last + symmetric zero padding
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
    """
    Fuse:
        NCHW -> channels-last
        +
        symmetric zero padding

    y keeps logical PyTorch shape [N, C, padded_h, padded_w], but its physical
    strides are channels-last.  Padding zeros are written in this same pass.
    """
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
    """Return channels-last [N,C,H+2P,W+2P] with fused copy + zero padding."""
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
        triton.cdiv(
            batch * padded_h * padded_w,
            _FUSED_PAD_CL_BLOCK_HW,
        ),
        triton.cdiv(
            channels,
            _FUSED_PAD_CL_BLOCK_C,
        ),
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
# V3 fixed 3x3 / Cin64 / Cout32 / OW128 path
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def conv2d_forward_3x3s1_p1_m128_k64_n32_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    HAS_BIAS: tl.constexpr,
):
    """
    Exact V3 kernel for:
        X logical shape: [N, 64, 130, 130] channels-last, already p1 padded
        W packed HWIO:   [3, 3, 64, 32]
        Y logical shape: [N, 32, 128, 128] channels-last

    One program computes one full output row:
        A: [128, 64]
        B: [64, 32]
        C: [128, 32]

    The nine 3x3 taps are explicitly unrolled.  There is no K loop, no spatial
    bounds mask, and no CI/CO tail mask for this exact shape.
    """
    pid_row = tl.program_id(0)

    # Exactly 128 output rows per image.
    n = pid_row // 128
    oh = pid_row - n * 128

    ow = tl.arange(0, 128)
    ci = tl.arange(0, 64)
    co = tl.arange(0, 32)

    acc = tl.zeros((128, 32), dtype=tl.float32)

    # tap (0, 0)
    x00_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w
    )
    w00_ptrs = (
        packed_w_ptr
        + 0 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x00 = tl.load(x00_ptrs)
    w00 = tl.load(w00_ptrs)
    acc += tl.dot(x00, w00, allow_tf32=False)

    # tap (0, 1)
    x01_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w
    )
    w01_ptrs = (
        packed_w_ptr
        + 0 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x01 = tl.load(x01_ptrs)
    w01 = tl.load(w01_ptrs)
    acc += tl.dot(x01, w01, allow_tf32=False)

    # tap (0, 2)
    x02_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w
    )
    w02_ptrs = (
        packed_w_ptr
        + 0 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x02 = tl.load(x02_ptrs)
    w02 = tl.load(w02_ptrs)
    acc += tl.dot(x02, w02, allow_tf32=False)

    # tap (1, 0)
    x10_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w
    )
    w10_ptrs = (
        packed_w_ptr
        + 1 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x10 = tl.load(x10_ptrs)
    w10 = tl.load(w10_ptrs)
    acc += tl.dot(x10, w10, allow_tf32=False)

    # tap (1, 1)
    x11_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w
    )
    w11_ptrs = (
        packed_w_ptr
        + 1 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x11 = tl.load(x11_ptrs)
    w11 = tl.load(w11_ptrs)
    acc += tl.dot(x11, w11, allow_tf32=False)

    # tap (1, 2)
    x12_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w
    )
    w12_ptrs = (
        packed_w_ptr
        + 1 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x12 = tl.load(x12_ptrs)
    w12 = tl.load(w12_ptrs)
    acc += tl.dot(x12, w12, allow_tf32=False)

    # tap (2, 0)
    x20_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w
    )
    w20_ptrs = (
        packed_w_ptr
        + 2 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x20 = tl.load(x20_ptrs)
    w20 = tl.load(w20_ptrs)
    acc += tl.dot(x20, w20, allow_tf32=False)

    # tap (2, 1)
    x21_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w
    )
    w21_ptrs = (
        packed_w_ptr
        + 2 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x21 = tl.load(x21_ptrs)
    w21 = tl.load(w21_ptrs)
    acc += tl.dot(x21, w21, allow_tf32=False)

    # tap (2, 2)
    x22_ptrs = (
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w
    )
    w22_ptrs = (
        packed_w_ptr
        + 2 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    x22 = tl.load(x22_ptrs)
    w22 = tl.load(w22_ptrs)
    acc += tl.dot(x22, w22, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + co).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + co[None, :] * y_stride_c
        + oh * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(y_ptrs, acc)


def _special_3x3_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
) -> torch.Tensor:
    """
    V3 specialized forward for the exact 128x128 p1 benchmark family.
    """
    batch = input.shape[0]

    # p1 is materialized explicitly.  The convolution kernel therefore sees
    # a fully valid [N,64,130,130] channels-last input and needs no H/W masks.
    padded_input = _nchw_to_padded_channels_last(
        input,
        1,
        1,
    )

    packed_weight = _get_prepacked_weight(weight)

    output = torch.empty(
        (batch, 32, 128, 128),
        device=input.device,
        dtype=input.dtype,
        memory_format=torch.channels_last,
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else output

    # One program per (N, OH) row.  Each program produces all 128 output-width
    # positions and all 32 output channels.
    grid = (batch * 128,)

    conv2d_forward_3x3s1_p1_m128_k64_n32_kernel[grid](
        padded_input,
        packed_weight,
        output,
        bias_ptr,
        *padded_input.stride(),
        *packed_weight.stride(),
        *output.stride(),
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=2,
    )

    return output


# ---------------------------------------------------------------------------
# V4 fixed 3x3 extensions: p2 / valid
# ---------------------------------------------------------------------------


@libentry()
@triton.jit
def conv2d_forward_3x3s1_p2_main_m128_k64_n32_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    HAS_BIAS: tl.constexpr,
):
    """
    V4 p2 main block.

    Input is already padded and channels-last:
        X: [N, 64, 132, 132]
    Output:
        Y: [N, 32, 130, 130]

    One program computes ow=[0,127] for one (N, OH) row.
    """
    pid_row = tl.program_id(0)

    n = pid_row // 130
    oh = pid_row - n * 130

    ow = tl.arange(0, 128)
    ci = tl.arange(0, 64)
    co = tl.arange(0, 32)

    acc = tl.zeros((128, 32), dtype=tl.float32)

    # (0,0)
    x00 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w
    )
    w00 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x00, w00, allow_tf32=False)

    # (0,1)
    x01 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w
    )
    w01 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x01, w01, allow_tf32=False)

    # (0,2)
    x02 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w
    )
    w02 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x02, w02, allow_tf32=False)

    # (1,0)
    x10 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w
    )
    w10 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x10, w10, allow_tf32=False)

    # (1,1)
    x11 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w
    )
    w11 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x11, w11, allow_tf32=False)

    # (1,2)
    x12 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w
    )
    w12 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x12, w12, allow_tf32=False)

    # (2,0)
    x20 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w
    )
    w20 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x20, w20, allow_tf32=False)

    # (2,1)
    x21 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w
    )
    w21 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x21, w21, allow_tf32=False)

    # (2,2)
    x22 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w
    )
    w22 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x22, w22, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + co).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + co[None, :] * y_stride_c
        + oh * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(y_ptrs, acc)


@libentry()
@triton.jit
def conv2d_forward_3x3s1_p2_tail2_m16_k64_n32_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    HAS_BIAS: tl.constexpr,
):
    """
    Logical two-column p2 tail.

    tl.dot backends commonly prefer M>=16, so this uses an M16 accumulator and
    masks every lane except ow=128,129.  The extra lanes never load/store.
    """
    pid_row = tl.program_id(0)

    n = pid_row // 130
    oh = pid_row - n * 130

    lane = tl.arange(0, 16)
    ow = 128 + lane
    valid = ow < 130

    ci = tl.arange(0, 64)
    co = tl.arange(0, 32)

    acc = tl.zeros((16, 32), dtype=tl.float32)

    # Nine taps. Invalid M lanes load zero.
    x00 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w00 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x00, w00, allow_tf32=False)

    x01 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w01 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x01, w01, allow_tf32=False)

    x02 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w02 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x02, w02, allow_tf32=False)

    x10 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w10 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x10, w10, allow_tf32=False)

    x11 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w11 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x11, w11, allow_tf32=False)

    x12 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w12 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x12, w12, allow_tf32=False)

    x20 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w20 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x20, w20, allow_tf32=False)

    x21 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w21 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x21, w21, allow_tf32=False)

    x22 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w22 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x22, w22, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + co).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + co[None, :] * y_stride_c
        + oh * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(y_ptrs, acc, mask=valid[:, None])


def _special_3x3_p2_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
) -> torch.Tensor:
    """V4 p2: M128 main + logical 2-column tail."""
    batch = input.shape[0]

    padded_input = _nchw_to_padded_channels_last(
        input,
        2,
        2,
    )
    packed_weight = _get_prepacked_weight(weight)

    output = torch.empty(
        (batch, 32, 130, 130),
        device=input.device,
        dtype=input.dtype,
        memory_format=torch.channels_last,
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else output
    grid_rows = (batch * 130,)

    conv2d_forward_3x3s1_p2_main_m128_k64_n32_kernel[grid_rows](
        padded_input,
        packed_weight,
        output,
        bias_ptr,
        *padded_input.stride(),
        *packed_weight.stride(),
        *output.stride(),
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=2,
    )

    conv2d_forward_3x3s1_p2_tail2_m16_k64_n32_kernel[grid_rows](
        padded_input,
        packed_weight,
        output,
        bias_ptr,
        *padded_input.stride(),
        *packed_weight.stride(),
        *output.stride(),
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=2,
    )

    return output


@libentry()
@triton.jit
def conv2d_forward_3x3s1_valid_m64_k64_n32_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    HAS_BIAS: tl.constexpr,
):
    """
    V4 valid path.

    X: [N,64,128,128] channels-last
    Y: [N,32,126,126] channels-last

    Two M64 programs cover each output row:
        tile 0: ow=[0,63]
        tile 1: ow=[64,127], with ow>=126 masked
    """
    pid = tl.program_id(0)

    tile = pid % 2
    row_id = pid // 2

    n = row_id // 126
    oh = row_id - n * 126

    ow = tile * 64 + tl.arange(0, 64)
    valid = ow < 126

    ci = tl.arange(0, 64)
    co = tl.arange(0, 32)

    acc = tl.zeros((64, 32), dtype=tl.float32)

    x00 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w00 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x00, w00, allow_tf32=False)

    x01 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w01 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x01, w01, allow_tf32=False)

    x02 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 0) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w02 = tl.load(
        packed_w_ptr
        + 0 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x02, w02, allow_tf32=False)

    x10 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w10 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x10, w10, allow_tf32=False)

    x11 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w11 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x11, w11, allow_tf32=False)

    x12 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 1) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w12 = tl.load(
        packed_w_ptr
        + 1 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x12, w12, allow_tf32=False)

    x20 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 0) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w20 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 0 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x20, w20, allow_tf32=False)

    x21 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 1) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w21 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 1 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x21, w21, allow_tf32=False)

    x22 = tl.load(
        x_ptr
        + n * x_stride_n
        + ci[None, :] * x_stride_c
        + (oh + 2) * x_stride_h
        + (ow[:, None] + 2) * x_stride_w,
        mask=valid[:, None],
        other=0.0,
    )
    w22 = tl.load(
        packed_w_ptr
        + 2 * pw_stride_h
        + 2 * pw_stride_w
        + ci[:, None] * pw_stride_i
        + co[None, :] * pw_stride_o
    )
    acc += tl.dot(x22, w22, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + co).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + co[None, :] * y_stride_c
        + oh * y_stride_h
        + ow[:, None] * y_stride_w
    )
    tl.store(y_ptrs, acc, mask=valid[:, None])


def _special_3x3_valid_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
) -> torch.Tensor:
    """V4 valid: M64 + M64, with the second tile masking two lanes."""
    batch = input.shape[0]

    # pad=0 makes this a pure NCHW -> channels-last conversion.
    cl_input = _nchw_to_padded_channels_last(
        input,
        0,
        0,
    )
    packed_weight = _get_prepacked_weight(weight)

    output = torch.empty(
        (batch, 32, 126, 126),
        device=input.device,
        dtype=input.dtype,
        memory_format=torch.channels_last,
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else output

    # 2 M64 tiles for every (N, OH) row.
    grid = (batch * 126 * 2,)

    conv2d_forward_3x3s1_valid_m64_k64_n32_kernel[grid](
        cl_input,
        packed_weight,
        output,
        bias_ptr,
        *cl_input.stride(),
        *packed_weight.stride(),
        *output.stride(),
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=2,
    )

    return output


# ---------------------------------------------------------------------------
# Packed implicit-GEMM convolution
# ---------------------------------------------------------------------------

# V5 targeted search space:
#   * retain the conservative V1-V4 candidates
#   * add N16 tiles for the large Cout=16 5x5 cases
#   * add low-stage N64 tiles for the large Cout=64 5x5 cases
#   * still avoid M64xN64 / larger 4096+ element fp32 accumulators
_PACKED_FORWARD_CONFIGS = [
    # ------------------------------------------------------------------
    # Baseline / V1-V4 candidates retained for regression safety.
    # ------------------------------------------------------------------
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 32,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 32,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 64,
            "BLOCK_CO": 32,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 64,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 32,
            "BLOCK_CI": 64,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 64,
            "BLOCK_CO": 32,
            "BLOCK_CI": 64,
        },
        num_warps=4,
        num_stages=2,
    ),
    # ------------------------------------------------------------------
    # V5: Cout=16 targeted candidates.
    #
    # The remaining large "same" benchmark has Cout=16.  BLOCK_CO=32
    # computes/holds twice as many output channels as are actually needed.
    # N16 candidates reduce accumulator/register footprint substantially.
    # ------------------------------------------------------------------
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 16,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 64,
            "BLOCK_CO": 16,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 64,
            "BLOCK_CO": 16,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 128,
            "BLOCK_CO": 16,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 16,
            "BLOCK_CI": 64,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 64,
            "BLOCK_CO": 16,
            "BLOCK_CI": 64,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 128,
            "BLOCK_CO": 16,
            "BLOCK_CI": 64,
        },
        num_warps=4,
        num_stages=1,
    ),
    # ------------------------------------------------------------------
    # V5: Cout=64 targeted candidates.
    #
    # Avoid M64xN64 (4096 fp32 accumulators).  The new stage-1 variants
    # intentionally favor lower register/pipeline pressure.
    # ------------------------------------------------------------------
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 16,
            "BLOCK_CO": 64,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 64,
            "BLOCK_CI": 16,
        },
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {
            "BLOCK_NI_HO_WO": 32,
            "BLOCK_CO": 64,
            "BLOCK_CI": 32,
        },
        num_warps=4,
        num_stages=1,
    ),
]


@libentry()
@triton.autotune(
    configs=_PACKED_FORWARD_CONFIGS,
    key=[
        "batch",
        "channels_per_group",
        "in_h",
        "in_w",
        "out_channels",
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "pad_h",
        "pad_w",
    ],
)
@triton.jit
def conv2d_forward_packed_kernel(
    x_ptr,
    packed_w_ptr,
    y_ptr,
    bias_ptr,
    batch,
    in_h,
    in_w,
    out_channels,
    out_h,
    out_w,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    pw_stride_h,
    pw_stride_w,
    pw_stride_i,
    pw_stride_o,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    channels_per_group: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHECK_SPATIAL_BOUNDS: tl.constexpr,
    BLOCK_NI_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Packed-weight implicit GEMM.

    A: [BLOCK_M, BLOCK_K] from input
    B: [BLOCK_K, BLOCK_N] from packed HWIO weight
    C: [BLOCK_M, BLOCK_N] output

    V2 large-5x5 path gives this kernel a pre-padded channels-last input:
      * x_stride_c == 1
      * pad_h == pad_w == 0
      * CHECK_SPATIAL_BOUNDS == False

    Therefore all per-tap H/W range checks are compiled out on that route.
    """
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)

    m = pid_m * BLOCK_NI_HO_WO + tl.arange(0, BLOCK_NI_HO_WO)
    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    out_hw = out_h * out_w
    n = m // out_hw
    hw = m - n * out_hw
    oh = hw // out_w
    ow = hw - oh * out_w

    acc = tl.zeros(
        (BLOCK_NI_HO_WO, BLOCK_CO),
        dtype=tl.float32,
    )

    k_blocks: tl.constexpr = (channels_per_group + BLOCK_CI - 1) // BLOCK_CI

    for r in range(kernel_h * kernel_w * k_blocks):
        kb = r % k_blocks
        tap = r // k_blocks
        kh = tap // kernel_w
        kw = tap - kh * kernel_w

        ci = kb * BLOCK_CI + tl.arange(0, BLOCK_CI)

        ih = oh * stride_h + kh - pad_h
        iw = ow * stride_w + kw - pad_w

        x_ptrs = (
            x_ptr
            + n[:, None] * x_stride_n
            + ci[None, :] * x_stride_c
            + ih[:, None] * x_stride_h
            + iw[:, None] * x_stride_w
        )

        w_ptrs = (
            packed_w_ptr
            + kh * pw_stride_h
            + kw * pw_stride_w
            + ci[:, None] * pw_stride_i
            + co[None, :] * pw_stride_o
        )

        base_x_mask = (n < batch)[:, None] & (ci < channels_per_group)[None, :]

        if CHECK_SPATIAL_BOUNDS:
            x_mask = (
                base_x_mask
                & (ih >= 0)[:, None]
                & (ih < in_h)[:, None]
                & (iw >= 0)[:, None]
                & (iw < in_w)[:, None]
            )
        else:
            x_mask = base_x_mask

        w_mask = (ci < channels_per_group)[:, None] & (co < out_channels)[None, :]

        x_tile = tl.load(
            x_ptrs,
            mask=x_mask,
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=w_mask,
            other=0.0,
        )

        acc += tl.dot(
            x_tile,
            w_tile,
            allow_tf32=False,
        )

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + co,
            mask=co < out_channels,
            other=0.0,
        ).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = (
        y_ptr
        + n[:, None] * y_stride_n
        + co[None, :] * y_stride_c
        + oh[:, None] * y_stride_h
        + ow[:, None] * y_stride_w
    )

    y_mask = (n < batch)[:, None] & (co < out_channels)[None, :]

    tl.store(
        y_ptrs,
        acc,
        mask=y_mask,
    )


def _packed_conv2d_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias,
    stride,
    padding,
    dilation,
):
    """
    Execute V1 packed path or V2 fused-CL 5x5 path.

    Original output H/W are always computed from the user's original input and
    padding.  If V2 preprocessing is selected, padding is materialized in the
    channels-last input and the convolution itself uses effective padding=0.
    """
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
        # Defensive fallback; dispatch should already have rejected this.
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

    convert_input_to_cl = _should_convert_input_to_channels_last(
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

    if convert_input_to_cl:
        # One Triton preprocessing launch performs both layout conversion and
        # symmetric zero-padding.  pad=0 is also supported and becomes a pure
        # NCHW->channels-last copy.
        forward_input = _nchw_to_padded_channels_last(
            input,
            pad_h,
            pad_w,
        )
        forward_in_h = in_h + 2 * pad_h
        forward_in_w = in_w + 2 * pad_w
        forward_pad_h = 0
        forward_pad_w = 0

        # Because every original padding value is now explicitly present in the
        # input buffer, every receptive field for the original output shape is
        # spatially valid.  Compile all spatial checks out of the conv kernel.
        check_spatial_bounds = False

    # Match the MThreads optimization idea: CO is the N dimension of the
    # implicit GEMM, so channels-last output makes CO stores contiguous.
    #
    # For V1 3x3 we keep NCHW output to isolate V2 behavior and avoid changing
    # the already-measured first version.
    if convert_input_to_cl:
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
        triton.cdiv(
            batch * out_h * out_w,
            meta["BLOCK_NI_HO_WO"],
        ),
        triton.cdiv(
            out_channels,
            meta["BLOCK_CO"],
        ),
    )

    has_bias = bias is not None
    # When HAS_BIAS=False, this pointer is specialized away and never loaded.
    bias_ptr = bias if has_bias else output

    conv2d_forward_packed_kernel[grid](
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
        HAS_BIAS=has_bias,
        CHECK_SPATIAL_BOUNDS=check_spatial_bounds,
    )

    return output


def conv2d_would_use_special_3x3(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether this call takes the V3 fixed 3x3 M128/K64/N32 path."""
    return _should_use_special_3x3(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )


def conv2d_would_use_special_3x3_p2(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether V4 p2 M128+tail2 is selected."""
    return _should_use_special_3x3_p2(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )


def conv2d_would_use_special_3x3_valid(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether V4 valid M64+M64(masked) is selected."""
    return _should_use_special_3x3_valid(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )


def conv2d_v5_targeted_autotune_family(weight: torch.Tensor) -> str:
    """
    Describe the V5 packed-kernel autotune focus for this weight.

    This helper is only for benchmark/debug output; Triton's autotuner still
    sees the full retained candidate list and selects the measured winner.
    """
    if weight.ndim != 4:
        return "generic"

    out_channels = int(weight.shape[0])

    if out_channels == 16:
        return "Cout16: prioritize BLOCK_CO=16, M=32/64/128, K=32/64"

    if out_channels == 64:
        return "Cout64: prioritize N64 with M=16/32, K=16/32, stages=1"

    return "baseline packed autotune candidates"


# ---------------------------------------------------------------------------
# Debug helpers for benchmark verification
# ---------------------------------------------------------------------------


def conv2d_would_use_packed_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether this call is eligible for the optimized packed route."""
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
    """Report whether V2 will run fused NCHW->CL(+padding) preprocessing."""
    if input.ndim != 4 or weight.ndim != 4:
        return False

    if input.dtype not in (torch.float16, torch.bfloat16):
        return False

    resolved_padding = _resolve_fast_padding(
        input,
        weight,
        stride,
        padding,
        dilation,
    )
    if resolved_padding is None:
        return False

    stride_h, stride_w = _normalize_2d(stride, "stride")
    dilation_h, dilation_w = _normalize_2d(dilation, "dilation")
    pad_h, pad_w = resolved_padding

    batch, _, in_h, in_w = input.shape
    _, channels_per_group, kernel_h, kernel_w = weight.shape

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

    return _should_use_packed_weight(
        input,
        weight,
        None,
        stride,
        padding,
        dilation,
        groups,
    ) and _should_convert_input_to_channels_last(
        input,
        batch,
        out_h,
        out_w,
        channels_per_group,
        kernel_h,
        kernel_w,
        groups,
    )


def conv2d_would_use_fused_nchw_cl_padding(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
) -> bool:
    """Report whether V2 performs a layout conversion that also adds padding."""
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
    return (pad_h > 0 or pad_w > 0) and conv2d_would_convert_input_to_channels_last(
        input,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


# ---------------------------------------------------------------------------
# Public backend entry
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
    THead conv2d entry.

    Route A -- V4/V3 specialized 3x3 family:
        p1 / OW128:
            -> fused CL+p1 -> M128/K64/N32 fixed nine-tap
        p2 / OW130:
            -> fused CL+p2 -> M128 main + logical tail2
        valid / OW126:
            -> pure NCHW->CL -> M64 + M64(masked)
        all use packed HWIO and fixed nine-tap kernels

    Route B -- V2:
        large dense FP16/BF16 5x5
        -> fused NCHW->channels-last(+padding)
        -> packed HWIO implicit GEMM
        -> no spatial bounds checks

    Route C -- V1:
        other eligible large FP16/BF16 3x3/5x5
        -> packed HWIO implicit GEMM

    Route D:
        everything else
        -> generic FlagGems conv2d
    """
    if _should_use_special_3x3_p2(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS_THEAD CONV2D V4_3X3_P2_M128_TAIL2")
        return _special_3x3_p2_forward(
            input,
            weight,
            bias,
        )

    if _should_use_special_3x3_valid(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS_THEAD CONV2D V4_3X3_VALID_M64_M64")
        return _special_3x3_valid_forward(
            input,
            weight,
            bias,
        )

    if _should_use_special_3x3(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    ):
        logger.debug("GEMS_THEAD CONV2D V3_3X3_M128_K64_N32")
        return _special_3x3_forward(
            input,
            weight,
            bias,
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
        resolved_padding = _resolve_fast_padding(
            input,
            weight,
            stride,
            padding,
            dilation,
        )
        pad_h, pad_w = resolved_padding

        stride_h, stride_w = _normalize_2d(stride, "stride")
        dilation_h, dilation_w = _normalize_2d(dilation, "dilation")
        batch, channels_per_group, in_h, in_w = input.shape
        _, _, kernel_h, kernel_w = weight.shape

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

        use_cl = _should_convert_input_to_channels_last(
            input,
            batch,
            out_h,
            out_w,
            channels_per_group,
            kernel_h,
            kernel_w,
            groups,
        )

        if use_cl:
            logger.debug("GEMS_THEAD CONV2D V2_FUSED_NCHW_CL_PAD_PACKED")
        else:
            logger.debug("GEMS_THEAD CONV2D V1_PACKED_WEIGHT")

        return _packed_conv2d_forward(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
        )

    logger.debug("GEMS_THEAD CONV2D GENERIC_FALLBACK")
    return _generic_conv2d(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
