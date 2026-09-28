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

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::int_repr materializes the raw integer storage of an affine-quantized
# tensor: quint8 -> uint8, qint8 -> int8, qint32 -> int32. The timed work is a
# read plus an equally sized write, so the benchmark dtype is the quantized
# input dtype. The quantizer runs outside the timed region because it only
# accepts a float32 source ("Quantize only works on Float Tensor").
_QUANTIZED_TO_INT_DTYPE = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}
QUANTIZED_DTYPES = list(_QUANTIZED_TO_INT_DTYPE)

# Six performance-relevant scales; the largest holds 2**24 elements, i.e. up to
# 64 MiB for a qint32 input plus 64 MiB of int32 output.
INT_REPR_MAIN_SHAPES = [
    (2**24,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 128, 64, 128),
]

# A rank-0 and a zero-sized input are natively valid quantized tensors on the
# device and are outside the scales above; both overloads list and replay them,
# and int_repr.out accepts a device buffer for the empty result.
INT_REPR_EXTRA_SHAPES = [
    (),
    (0,),
]

INT_REPR_SHAPES = INT_REPR_MAIN_SHAPES + INT_REPR_EXTRA_SHAPES

_LAYOUTS = (
    "contiguous",
    "transposed",
    "offset_slice",
    "strided_last",
    "expanded",
    "channels_last",
    "per_channel",
)

# Per-channel scales and zero points are float64 / int64 metadata, so those
# cases are only built where both dtypes exist. This is an import-time
# capability gate, not a size limit.
_PER_CHANNEL_SUPPORTED = (
    flag_gems.runtime.device.support_fp64 and flag_gems.runtime.device.support_int64
)

_SCALE_CYCLE = (1.0, 2.0**20, 2.0**-20, 0.5, 255.0 / 150.0, 1e-3)
_FRACTION_CYCLE = (0.0, 1.0, 0.5)
_CHANNEL_SCALE_CYCLE = (2.0**-8, 2.0**-4, 0.5, 1.0, 2.0, 8.0)


def _int_dtype(dtype):
    return _QUANTIZED_TO_INT_DTYPE[dtype]


def _per_tensor_qparams(dtype, index):
    # The extremes and the middle of the storage dtype's legal zero-point
    # range, so no unsigned value is ever handed to a signed dtype.
    info = torch.iinfo(_int_dtype(dtype))
    span = info.max - info.min
    scale = _SCALE_CYCLE[index % len(_SCALE_CYCLE)]
    fraction = _FRACTION_CYCLE[index % len(_FRACTION_CYCLE)]
    return scale, int(info.min + round(fraction * span))


def _channel_qparams(channels, dtype):
    info = torch.iinfo(_int_dtype(dtype))
    span = info.max - info.min
    scales = [
        _CHANNEL_SCALE_CYCLE[i % len(_CHANNEL_SCALE_CYCLE)] for i in range(channels)
    ]
    zero_points = [
        int(info.min + (span * i) // max(channels - 1, 1)) for i in range(channels)
    ]
    return scales, zero_points


def _layout_shape(shape, layout):
    """Exact shape produced by _view for this layout."""
    if layout in ("contiguous", "channels_last", "per_channel"):
        return tuple(shape)
    if layout == "transposed":
        return (shape[1], shape[0], *shape[2:])
    if layout == "offset_slice":
        return (shape[0] - 1, *shape[1:])
    if layout == "strided_last":
        return (*shape[:-1], (shape[-1] + 1) // 2)
    if layout == "expanded":
        return (4, *shape[1:])
    raise ValueError(f"unknown layout {layout!r}")


def _view(inp, layout):
    if layout in ("contiguous", "per_channel"):
        return inp
    if layout == "transposed":
        return inp.transpose(0, 1)
    if layout == "offset_slice":
        return inp[1:]
    if layout == "strided_last":
        return inp[..., ::2]
    if layout == "expanded":
        return inp[:1].expand(4, *inp.shape[1:])
    if layout == "channels_last":
        return inp.to(memory_format=torch.channels_last)
    raise ValueError(f"unknown layout {layout!r}")


def _applicable_layouts(shape):
    layouts = ["contiguous"]
    if len(shape) >= 2 and shape[0] >= 1:
        layouts.extend(("transposed", "expanded"))
    if len(shape) >= 1 and shape[0] >= 1:
        layouts.append("offset_slice")
    if len(shape) >= 1:
        layouts.append("strided_last")
    if len(shape) == 4 and all(dim > 0 for dim in shape):
        layouts.append("channels_last")
    if _PER_CHANNEL_SUPPORTED and len(shape) >= 1 and all(dim > 0 for dim in shape):
        layouts.append("per_channel")
    return layouts


def _validated_shape(shape):
    if isinstance(shape, bool) or not isinstance(shape, (tuple, list)):
        raise ValueError(f"shape metadata must be a tuple of integers, got {shape!r}")
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"shape dimension must be a plain integer, got {dim!r}")
        if dim < 0:
            raise ValueError(f"shape dimension must not be negative, got {dim!r}")
        dims.append(dim)
    return tuple(dims)


def _validate(plan, dtype):
    shape, layout, scale, zero_point, axis = plan.builder_args
    shape = _validated_shape(shape)
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    if tuple(plan.shape["input"]) != _layout_shape(shape, layout):
        raise ValueError(
            f"plan metadata {tuple(plan.shape['input'])} does not describe the "
            f"{layout} view of {shape}"
        )
    if layout == "per_channel":
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError(f"per-channel axis must be an integer, got {axis!r}")
        if not 0 <= axis < len(shape):
            raise ValueError(f"per-channel axis {axis} is out of range for {shape}")
        if not isinstance(scale, list) or not isinstance(zero_point, list):
            raise ValueError("per-channel scales and zero points must be lists")
        channels = shape[axis]
        if len(scale) != channels or len(zero_point) != channels:
            raise ValueError(
                f"per-channel metadata must hold {channels} entries for axis {axis}"
            )
        for value in scale:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"per-channel scale must be numeric, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"per-channel scale must be finite and positive, got {value!r}"
                )
    else:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError(f"scale must be numeric, got {scale!r}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and positive, got {scale!r}")
    info = torch.iinfo(_int_dtype(dtype))
    zero_points = zero_point if isinstance(zero_point, list) else [zero_point]
    for value in zero_points:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"zero point must be an integer, got {value!r}")
        if not info.min <= value <= info.max:
            raise ValueError(
                f"zero point {value} is outside the {info.min}..{info.max} range of "
                f"{_int_dtype(dtype)}"
            )
    return shape


def _case_fn(shape, dtype):
    for index, layout in enumerate(_applicable_layouts(shape)):
        if layout == "per_channel":
            axis = len(shape) - 1
            scales, zero_points = _channel_qparams(shape[axis], dtype)
            params = {
                "layout": layout,
                "qscheme": "per_channel_affine",
                "axis": axis,
                "scale": scales,
                "zero_point": zero_points,
            }
            builder_args = (tuple(shape), layout, scales, zero_points, axis)
        else:
            scale, zero_point = _per_tensor_qparams(dtype, sum(shape) + index)
            params = {
                "layout": layout,
                "qscheme": "per_tensor_affine",
                "scale": scale,
                "zero_point": zero_point,
            }
            builder_args = (tuple(shape), layout, scale, zero_point, None)
        yield base.BenchmarkCasePlan(
            shape={"input": list(_layout_shape(shape, layout))},
            params=params,
            builder_args=builder_args,
        )


def _payload(shape, device):
    # float32 payload arithmetic only: the quantizer rejects any other source
    # dtype and the per-tensor fixtures need no float64 support.
    return torch.empty(shape, dtype=torch.float32, device=device).uniform_(-1.0, 1.0)


def _quantized_input(plan, dtype, device):
    shape, layout, scale, zero_point, axis = plan.builder_args
    shape = _validate(plan, dtype)
    payload = _payload(shape, device)
    if layout == "per_channel":
        inp = torch.quantize_per_channel(
            payload,
            torch.tensor(scale, dtype=torch.float64, device=device),
            torch.tensor(zero_point, dtype=torch.int64, device=device),
            axis,
            dtype,
        )
    elif payload.numel() == 0:
        # Empty quantized inputs are built directly on the requested device, so
        # the empty scales exercise the accelerator and .out can use a device
        # buffer for the empty result.
        inp = torch._empty_affine_quantized(
            list(shape), scale=scale, zero_point=zero_point, dtype=dtype, device=device
        )
    else:
        inp = torch.quantize_per_tensor(payload, scale, zero_point, dtype)
    return _view(inp, layout)


def _build_inputs_fn(plan, dtype, device):
    inp = _quantized_input(plan, dtype, device)
    return inp, {}


def _build_inputs_fn_out(plan, dtype, device):
    inp = _quantized_input(plan, dtype, device)
    # The .out overload writes into a plain integer buffer of the storage dtype;
    # allocating it here keeps it out of the timed region.
    out = torch.empty(
        tuple(plan.shape["input"]), dtype=_int_dtype(dtype), device=device
    )
    return inp, {"out": out}


class IntReprBenchmark(OperatorBenchmark):
    """Two-phase benchmark for an operator that consumes a quantized tensor.

    The pointwise families and utils.generate_tensor_input build plain float or
    integer tensors, so the quantized input is produced by build_inputs_fn
    instead. set_shapes pins the default scale list while an explicitly supplied
    shape file still takes precedence; invalid custom rows raise instead of
    being replaced by the default shapes.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=INT_REPR_SHAPES)
        self.shapes = [_validated_shape(shape) for shape in self.shapes]


@pytest.mark.int_repr
def test_int_repr():
    bench = IntReprBenchmark(
        op_name="int_repr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.int_repr,
        gems_op=getattr(flag_gems, "int_repr", None),
        dtypes=QUANTIZED_DTYPES,
    )
    bench.run()


@pytest.mark.int_repr
def test_int_repr_out():
    bench = IntReprBenchmark(
        op_name="int_repr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.int_repr.out,
        gems_op=getattr(flag_gems, "int_repr", None),
        dtypes=QUANTIZED_DTYPES,
    )
    bench.run()
