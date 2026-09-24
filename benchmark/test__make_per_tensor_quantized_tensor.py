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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base

# ``_make_per_tensor_quantized_tensor`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register the markers directly on the MarkGenerator so
# ``@pytest.mark._make_per_tensor_quantized_tensor`` and ``-m
# _make_per_tensor_quantized_tensor`` both work.
for _name in (
    "_make_per_tensor_quantized_tensor",
    "_make_per_tensor_quantized_tensor_out",
):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int
# zero_point) -> Tensor wraps an integer tensor into a per-tensor affine
# quantized tensor: the output dtype is derived from the input dtype via
# toQIntType (uint8 -> quint8, int8 -> qint8, int32 -> qint32) and the data path
# is a pure copy, so the benchmark measures copy bandwidth plus output
# allocation. Only these three integer input dtypes are accepted.
MAKE_PERTENSOR_INPUT_DTYPES = [torch.uint8, torch.int8, torch.int32]
_QUANT_DTYPE = {
    torch.uint8: torch.quint8,
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
}

# Fixed quantization parameters used for every benchmark case (the op stores
# them verbatim, so they do not affect the data path).
SCALE = 0.1
ZERO_POINT = 0

# The default shape set contains a 1-G-element 1-D tensor whose int32 input +
# output would need ~8 GiB and OOM on busy GPUs. Use allocation-friendly shapes
# instead, spread over 1-D..4-D and capped at 2**24 elements (64 MiB int32 per
# tensor, so a case needs ~128 MiB for input + output).
MAKE_PERTENSOR_SHAPES = [
    (2**24,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 128, 64, 128),
    (8, 256, 256, 32),
]


def _make_input(shape, dtype, device):
    info = torch.iinfo(dtype)
    return torch.randint(info.min, info.max, shape, dtype=dtype, device=device)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"scale": SCALE, "zero_point": ZERO_POINT},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = _make_input(shape, dtype, device)
    return inp, {
        "scale": plan.params["scale"],
        "zero_point": plan.params["zero_point"],
    }


def _build_inputs_fn_out(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = _make_input(shape, dtype, device)
    # The .out overload writes into an existing quantized tensor whose dtype is
    # the derived quantized dtype; allocate it with the same shape and different
    # qparams so the overwrite is observable (allocation is not timed).
    out = torch.ops.aten._empty_affine_quantized(
        shape, dtype=_QUANT_DTYPE[dtype], device=device, scale=1.0, zero_point=0
    )
    return inp, {
        "scale": plan.params["scale"],
        "zero_point": plan.params["zero_point"],
        "out": out,
    }


class MakePerTensorQuantizedTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark that supplies the fixed scale/zero_point.

    aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int
    zero_point) -> Tensor needs scalar qparams alongside the tensor, which the
    pointwise families do not supply, so the case builder and input builder
    forward them explicitly. The output is a quantized tensor of the same shape
    (dtype derived from the input dtype), so each case needs input + output (2x
    one tensor's memory). set_shapes pins the allocation-friendly shape list
    instead of the default 1-G-element shapes.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=MAKE_PERTENSOR_SHAPES)


@pytest.mark._make_per_tensor_quantized_tensor
def test__make_per_tensor_quantized_tensor():
    bench = MakePerTensorQuantizedTensorBenchmark(
        op_name="_make_per_tensor_quantized_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._make_per_tensor_quantized_tensor,
        gems_op=getattr(flag_gems, "_make_per_tensor_quantized_tensor", None),
        dtypes=MAKE_PERTENSOR_INPUT_DTYPES,
    )
    bench.run()


@pytest.mark._make_per_tensor_quantized_tensor_out
def test__make_per_tensor_quantized_tensor_out():
    bench = MakePerTensorQuantizedTensorBenchmark(
        op_name="_make_per_tensor_quantized_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._make_per_tensor_quantized_tensor.out,
        gems_op=getattr(flag_gems, "_make_per_tensor_quantized_tensor", None),
        dtypes=MAKE_PERTENSOR_INPUT_DTYPES,
    )
    bench.run()
