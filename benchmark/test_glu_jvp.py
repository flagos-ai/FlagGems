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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

_SUPPORT_BF16 = flag_gems.runtime.device.support_bf16

# Timed family follows consts.FLOAT_DTYPES with the bf16 capability flag applied;
# float64 correctness is covered in tests/test_glu_jvp.py.
_BENCH_DTYPES = [
    dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16 or _SUPPORT_BF16
]

# Default shapes mirror the 'glu:' entry of core_shapes.yaml; a caller-supplied
# shape file still wins.
GLU_JVP_SHAPES = [
    (4, 8, 512, 128),
    (4, 8, 1024, 128),
    (4, 8, 2048, 128),
    (4, 8, 3072, 128),
    (4, 8, 4096, 128),
]


def _benchmark_dims(shape):
    """halving dims benchmarked for shape: the last dim, plus dim 0 if it exists.

    A primal extent of zero is a valid native workload (the output is empty),
    so no shape is filtered out here and every requested case is timed.
    """
    return [-1] + ([0] if len(shape) > 1 else [])


def _case_fn(shape, dtype):
    del dtype
    for dim in _benchmark_dims(shape):
        glu_shape = list(shape)
        glu_shape[dim] = shape[dim] // 2
        yield base.BenchmarkCasePlan(
            shape={"x": list(shape), "glu": glu_shape},
            params={"dim": dim},
            builder_args=(tuple(shape), dim),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    glu_shape = list(shape)
    glu_shape[dim] = shape[dim] // 2
    glu = utils.generate_tensor_input(tuple(glu_shape), dtype, device)
    x = utils.generate_tensor_input(shape, dtype, device)
    dx = utils.generate_tensor_input(shape, dtype, device)
    return glu, x, dx, dim


class GluJvpBenchmark(OperatorBenchmark):
    """Two-phase benchmark defaulting to the glu shapes of core_shapes.yaml."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=GLU_JVP_SHAPES)


@pytest.mark.glu_jvp
def test_glu_jvp():
    bench = GluJvpBenchmark(
        op_name="glu_jvp",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.glu_jvp,
        gems_op=getattr(flag_gems, "glu_jvp", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
