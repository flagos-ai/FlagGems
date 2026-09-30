# Copyright 2025, The FlagOS Contributors.
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

pytestmark = [
    pytest.mark.fbgemm_linear_fp16_weight,
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

# CPU-only FBGEMM API: the activation and the bias are dense CPU tensors and the
# weight is the opaque handle returned by aten::fbgemm_pack_gemm_matrix_fp16, so
# the builders allocate CPU operands directly and ignore the framework device.
_DTYPES = [torch.float32]
_DEVICE = torch.device("cpu")
_BIAS_FORMS = ("broadcast", "full")
_MAX_FEATURES = 1024
_FOLD_WIDTH = 1024
_EXTRA_SHAPES = [
    (2, 19, 7),
    (16, 128, 60),
    (16, 7, 57, 32, 29),
    (256, 1),
    (2, 4096),
]


def _native_geometry(shape):
    # Map a caller shape onto a native-valid GEMM without changing its numel: the op
    # needs a rank >= 2 activation whose last dim is the packed weight inner dim, so
    # a 0 / 1-D shape is folded into (rows, width) using a gcd divisor of the numel
    # (floor division would drop elements) while an empty input stays empty. Only
    # the weight output-feature count is capped, at _MAX_FEATURES rows, so the packed
    # weight stays small; the activation allocation is untouched.
    shape = tuple(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    if len(shape) >= 2:
        act_shape = shape
    elif numel == 0:
        act_shape = (0, 1)
    else:
        width = math.gcd(numel, _FOLD_WIDTH)
        act_shape = (numel // width, width)
    inner = act_shape[-1]
    return act_shape, (min(inner, _MAX_FEATURES) if inner > 0 else 1, inner)


def _case_fn(shape, dtype):
    del dtype
    act_shape, weight_shape = _native_geometry(shape)
    for bias_form in _BIAS_FORMS:
        yield base.BenchmarkCasePlan(
            shape={"input": act_shape, "weight": weight_shape},
            params={"bias": bias_form},
            builder_args=(act_shape, weight_shape, bias_form),
        )


def _build_inputs_fn(plan, dtype, device):
    del device
    act_shape, weight_shape, bias_form = plan.builder_args
    inp = torch.randn(act_shape, dtype=dtype, device=_DEVICE)
    weight = torch.randn(weight_shape, dtype=torch.float32, device=_DEVICE)
    packed = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
    bias = torch.randn(
        (1,) if bias_form == "broadcast" else (weight_shape[0],),
        dtype=dtype,
        device=_DEVICE,
    )
    # Flat positional arguments: unpack_to_args_kwargs flattens this tuple and
    # there is no trailing kwargs dict.
    return inp, packed, bias


class FbgemmLinearFp16WeightBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Keep the shared grid (and any user shape file), then union the native-valid
        # extra geometries instead of replacing the defaults.
        super().set_shapes(shape_file_path)
        existing = {tuple(shape) for shape in self.shapes}
        for shape in _EXTRA_SHAPES:
            if tuple(shape) not in existing:
                self.shapes.append(shape)
                existing.add(tuple(shape))


@pytest.mark.fbgemm_linear_fp16_weight
def test_fbgemm_linear_fp16_weight():
    bench = FbgemmLinearFp16WeightBenchmark(
        op_name="fbgemm_linear_fp16_weight",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_linear_fp16_weight,
        gems_op=getattr(flag_gems, "fbgemm_linear_fp16_weight", None),
        dtypes=_DTYPES,
    )
    bench.run()
