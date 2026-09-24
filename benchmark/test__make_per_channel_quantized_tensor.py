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

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base

# ``_make_per_channel_quantized_tensor`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._make_per_channel_quantized_tensor`` and ``-m
# _make_per_channel_quantized_tensor`` both work.
for _name in (
    "_make_per_channel_quantized_tensor",
    "_make_per_channel_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor
# zero_point, int axis) -> Tensor wraps an integer storage tensor into a
# per-channel affine quantized tensor. The output dtype is derived from the
# input dtype (uint8 -> quint8, int8 -> qint8, int32 -> qint32) and the data
# path is a pure bit copy, so the benchmark measures copy bandwidth plus the
# per-channel output allocation. Only these three integer input dtypes are
# accepted, so the dtype set is local rather than consts.FLOAT_DTYPES.
STORAGE_DTYPES = [torch.uint8, torch.int8, torch.int32]
_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

# (shape, axis) pairs. The axis must stay small because the per-channel
# metadata has one element per slice along it, while the element count drives
# the copy cost. The default shape set contains a 1-G-element 1-D tensor whose
# int32 input + output would need ~8 GiB and OOM on busy GPUs, so these
# allocation-friendly shapes (<= 2**26 elements, <= 256 MiB int32 per tensor)
# are used instead.
SHAPE_AXIS = [
    ((2**20,), 0),  # 1M channels, 1M elements
    ((1024, 1024), 0),  # 1024 channels, 1M elements
    ((4096, 4096), 0),  # 4096 channels, 16M elements
    ((64, 512, 512), 0),  # 64 channels, 16M elements
    ((16, 128, 64, 1280), 0),  # 16 channels, 16.7M elements
    ((8, 512, 512, 32), 0),  # 8 channels, 67M elements
]


def _make_input(shape, dtype, device):
    info = torch.iinfo(dtype)
    return torch.randint(info.min, info.max + 1, shape, dtype=dtype, device=device)


def _make_metadata(num_channels, device):
    # Per-channel metadata has one entry per slice along the axis; the op stores
    # it verbatim, so the values do not affect the timing of the copy path.
    scales = torch.rand(num_channels, dtype=torch.float32, device=device) + 0.1
    zero_points = torch.zeros(num_channels, dtype=torch.int64, device=device)
    return scales, zero_points


def _case_fn(shape, dtype):
    del dtype
    axis = 0
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"axis": axis},
        builder_args=(shape, axis),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, axis = plan.builder_args
    inp = _make_input(shape, dtype, device)
    scales, zero_points = _make_metadata(shape[axis], device)
    return inp, {
        "scale": scales,
        "zero_point": zero_points,
        "axis": plan.params["axis"],
    }


def _build_inputs_fn_out(plan, dtype, device):
    shape, axis = plan.builder_args
    inp = _make_input(shape, dtype, device)
    scales, zero_points = _make_metadata(shape[axis], device)
    # The .out overload writes into an existing per-channel quantized tensor
    # whose dtype is the derived quantized dtype; allocate it with the same shape
    # and different metadata so the overwrite is observable (allocation happens
    # in the builder and is not timed).
    num_channels = shape[axis]
    out = torch.ops.aten._empty_per_channel_affine_quantized(
        shape,
        scales=torch.full((num_channels,), 9.0, dtype=torch.float64, device=device),
        zero_points=torch.full((num_channels,), 9, dtype=torch.int64, device=device),
        axis=axis,
        dtype=_QUANT_DTYPE[dtype],
        device=device,
    )
    return inp, {
        "scale": scales,
        "zero_point": zero_points,
        "axis": plan.params["axis"],
        "out": out,
    }


class MakePerChannelQuantizedTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark that supplies the per-channel metadata.

    aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor
    zero_point, int axis) -> Tensor needs the axis plus two metadata tensors
    alongside the input, which the pointwise families do not supply, so the case
    builder and input builder forward them explicitly. The output is a quantized
    tensor of the same shape (dtype derived from the input dtype), so each case
    needs input + output (2x one tensor's memory).
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path, default_shapes=[shape for shape, _ in SHAPE_AXIS]
        )


@pytest.mark._make_per_channel_quantized_tensor
def test__make_per_channel_quantized_tensor():
    bench = MakePerChannelQuantizedTensorBenchmark(
        op_name="_make_per_channel_quantized_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._make_per_channel_quantized_tensor,
        # KernelGen installs the candidate through --override;
        # the direct attribute may not exist yet, so fall back to None (the
        # resolve step then picks up the override).
        gems_op=getattr(flag_gems, "_make_per_channel_quantized_tensor", None),
        dtypes=STORAGE_DTYPES,
    )
    bench.run()


@pytest.mark._make_per_channel_quantized_tensor_out
def test__make_per_channel_quantized_tensor_out():
    bench = MakePerChannelQuantizedTensorBenchmark(
        op_name="_make_per_channel_quantized_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._make_per_channel_quantized_tensor.out,
        gems_op=getattr(flag_gems, "_make_per_channel_quantized_tensor", None),
        dtypes=STORAGE_DTYPES,
    )
    bench.run()
