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
from .generated_operator_utils import OperatorBenchmark

# q_per_channel_scales returns stored per-channel quantizer metadata, so the default
# latency metrics describe the workload (byte throughput and FLOPs do not), the "dtype"
# axis is the quantized storage dtype, and every storage dtype runs through each
# per-channel qparams family it supports. Both call forms are measured.
#
# The per-channel axis is a schema parameter, so axes are derived per shape to exercise
# the first, middle and last dimensions of the shared default shapes. Rank-0, 1-D,
# zero-channel and empty shapes are added through the shared set_more_shapes() hook; a
# caller-supplied shape file still wins through the shared loader.

STORAGE_BYTES = {
    torch.qint8: torch.int8,
    torch.quint8: torch.uint8,
    torch.qint32: torch.int32,
}

# per_channel_affine keeps float64 scales with integer zero points (int32 for qint32
# storage); per_channel_affine_float_qparams keeps float32 for both and exists for
# 8-bit storage only.
FAMILIES_BY_STORAGE = {
    torch.qint8: ("affine", "float_qparams"),
    torch.quint8: ("affine", "float_qparams"),
    torch.qint32: ("affine",),
}

QPARAM_DTYPES = {
    "affine": (torch.float64, torch.int64),
    "float_qparams": (torch.float32, torch.float32),
}

RESULT_DTYPE = {"affine": torch.float64, "float_qparams": torch.float32}

STORAGE_DTYPES = [torch.qint8, torch.quint8, torch.qint32]

# Extra per-channel boundaries beyond the framework shape set: single channel, the
# remaining spec shapes, rank-0, empty storage, a zero-length channel axis and an empty
# tensor whose metadata is non-empty.
EXTRA_SHAPES = [
    (256,),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (),
    (0,),
    (4, 0, 3),
    (0, 5),
]

# (0, 5) has no channels on axis 0 but five on axis -1.
AXES_BY_SHAPE = {(0, 5): (0, -1)}


def _channels(shape, axis):
    # A rank-0 quantized tensor reports a single scale on axis 0.
    return 1 if len(shape) == 0 else shape[axis]


def _axes_for(shape):
    if len(shape) <= 1:
        return AXES_BY_SHAPE.get(shape, (0,))
    return AXES_BY_SHAPE.get(shape, tuple(sorted({0, len(shape) // 2, len(shape) - 1})))


def _quantized_input(shape, axis, storage_dtype, family, device):
    scales_dtype, zero_points_dtype = QPARAM_DTYPES[family]
    channels = _channels(shape, axis)
    scales = torch.arange(1, channels + 1, dtype=scales_dtype, device=device) * 0.5
    zero_points = (torch.arange(channels, dtype=torch.int32, device=device) % 7).to(
        zero_points_dtype
    )
    numel = 1
    for extent in shape:
        numel *= extent
    storage = (
        (torch.arange(numel, dtype=torch.int32, device=device) % 100)
        .to(STORAGE_BYTES[storage_dtype])
        .reshape(shape)
    )
    return torch.ops.aten._make_per_channel_quantized_tensor(
        storage, scales, zero_points, axis
    )


def _case_fn(shape, dtype):
    # Listing stays tensor-free: only JSON-compatible metadata is emitted here.
    shape = tuple(shape)
    for axis in _axes_for(shape):
        channels = _channels(shape, axis)
        for family in FAMILIES_BY_STORAGE[dtype]:
            yield base.BenchmarkCasePlan(
                shape={"input": shape},
                params={
                    "axis": axis,
                    "channels": channels,
                    "family": family,
                    "storage_dtype": str(dtype),
                },
                builder_args=(shape, axis, dtype, family),
            )


def _build_inputs_fn(plan, dtype, device):
    shape, axis, storage_dtype, family = plan.builder_args
    return (_quantized_input(shape, axis, storage_dtype, family, device),)


def _build_inputs_fn_out(plan, dtype, device):
    shape, axis, storage_dtype, family = plan.builder_args
    inp = _quantized_input(shape, axis, storage_dtype, family, device)
    # Allocation happens in the builder and is not timed. out is keyword-only in the
    # native schema, so it travels in the kwargs dict after the positional arguments.
    out_buf = torch.empty(
        _channels(shape, axis), dtype=RESULT_DTYPE[family], device=device
    )
    return inp, {"out": out_buf}


class QPerChannelScalesBenchmark(OperatorBenchmark):
    def set_more_shapes(self):
        # Extra per-channel boundaries merged by the shared loader at the comprehensive
        # level; a caller shape file still overrides the default shapes.
        return super().set_more_shapes() + list(EXTRA_SHAPES)


@pytest.mark.q_per_channel_scales
def test_q_per_channel_scales():
    bench = QPerChannelScalesBenchmark(
        op_name="q_per_channel_scales",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.q_per_channel_scales,
        gems_op=getattr(flag_gems, "q_per_channel_scales", None),
        dtypes=STORAGE_DTYPES,
    )
    bench.run()


@pytest.mark.q_per_channel_scales
def test_q_per_channel_scales_out():
    bench = QPerChannelScalesBenchmark(
        op_name="q_per_channel_scales",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.q_per_channel_scales.out,
        gems_op=getattr(flag_gems, "q_per_channel_scales", None),
        dtypes=STORAGE_DTYPES,
    )
    bench.run()
