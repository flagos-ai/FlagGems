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

import pytest
import torch

import flag_gems

from . import base

# aten::q_per_channel_zero_points only reads quantizer metadata, so the timed
# work is dispatch plus the channel count. The shared shape list (and any caller
# --shape_file) drives the cases unchanged: the operator schema imposes no size
# constraint beyond the channel axis, so shapes are not filtered or capped.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

# Both result families are timed: per_channel_affine stores int64 zero points,
# per_channel_affine_float_qparams stores float32 ones.
QSCHEMES = ["affine", "float_qparams"]

_OUT_DTYPE = {"affine": torch.int64, "float_qparams": torch.float32}


def _num_channels(shape, axis):
    # Rank 0 is a valid per-channel operand with a single channel on axis 0.
    return shape[axis] if shape else 1


def _axes(shape):
    if len(shape) <= 1:
        return (0,)
    return (0, -1)


def _make_quantized(shape, axis, qscheme, quantized_dtype, device):
    num_channels = _num_channels(shape, axis)
    scales = torch.ones(num_channels, dtype=torch.float64, device=device)
    if qscheme == "float_qparams":
        zero_points = torch.full(
            (num_channels,), 0.5, dtype=torch.float32, device=device
        )
    else:
        zero_points = torch.arange(num_channels, dtype=torch.int64, device=device)
    return torch.ops.aten._empty_per_channel_affine_quantized(
        tuple(shape),
        scales=scales,
        zero_points=zero_points,
        axis=axis,
        dtype=quantized_dtype,
        device=device,
    )


def _case_fn(shape, dtype):
    # Listing never allocates: the plans carry only JSON-compatible metadata,
    # including the advertised element dtype that the builders will use.
    for axis in _axes(shape):
        for qscheme in QSCHEMES:
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={"axis": axis, "qscheme": qscheme, "dtype": str(dtype)},
                builder_args=(tuple(shape), axis, qscheme),
            )


def _build_inputs_fn(plan, dtype, device):
    # generate_tensor_input cannot build a quantized tensor, so the input is
    # created here with the case dtype as its quantized element dtype.
    shape, axis, qscheme = plan.builder_args
    return _make_quantized(shape, axis, qscheme, dtype, device), {}


def _build_inputs_fn_out(plan, dtype, device):
    shape, axis, qscheme = plan.builder_args
    quantized = _make_quantized(shape, axis, qscheme, dtype, device)
    # The .out overload writes into (and returns) a buffer whose dtype follows
    # the qscheme: long for the integer scheme, float for the float one.
    out = torch.empty(
        _num_channels(shape, axis), dtype=_OUT_DTYPE[qscheme], device=device
    )
    return quantized, {"out": out}


class QPerChannelZeroPointsBenchmark(base.GenericBenchmark):
    """The shared shape list has no rank-0 entry, so the scalar operand (one
    channel on axis 0) is contributed here; the shared and caller-provided
    shapes are kept untouched."""

    def set_more_shapes(self):
        return super().set_more_shapes() + [(), (0, 3), (3, 0)]


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points():
    bench = QPerChannelZeroPointsBenchmark(
        op_name="q_per_channel_zero_points",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.q_per_channel_zero_points,
        gems_op=getattr(flag_gems, "q_per_channel_zero_points", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()


@pytest.mark.q_per_channel_zero_points
def test_q_per_channel_zero_points_out():
    bench = QPerChannelZeroPointsBenchmark(
        op_name="q_per_channel_zero_points",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.q_per_channel_zero_points.out,
        gems_op=getattr(flag_gems, "q_per_channel_zero_points", None),
        dtypes=QUANT_DTYPES,
    )
    bench.run()
