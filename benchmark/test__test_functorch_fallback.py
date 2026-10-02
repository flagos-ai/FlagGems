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

# aten::_test_functorch_fallback(self, other) copies 'self' and ignores 'other'.
# Its only kernels are CPU and Meta, so the benchmark runs on CPU tensors and
# declares device="cpu"; the shared core/comprehensive shape set is kept and only
# extended with the CPU-relevant sizes below.
_TFF_EXTRA_SHAPES = [
    (256,),
    (4096,),
    (2, 19, 7),
    (64, 64),
    (20, 320, 15),
    (1024, 1024),
    (16, 128, 64),
]

# Full native CPU dtype support: the nine required types plus float64 and bool.
_TFF_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
    torch.int16,
    torch.complex64,
    torch.complex128,
]


def _case_fn(shape, dtype):
    del dtype
    for other_shape in dict.fromkeys([tuple(shape), (1,)]):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape), "other": list(other_shape)},
            params={},
            builder_args=(shape, other_shape),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, other_shape = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    other = torch.empty(other_shape, dtype=dtype, device=device)
    return inp, other, {}


class FunctorchFallbackBenchmark(OperatorBenchmark):
    """Keeps every shared shape and adds the CPU-sized workloads below."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape) for shape in list(self.shapes) + _TFF_EXTRA_SHAPES
            )
        )


@pytest.mark.test_functorch_fallback
def test_test_functorch_fallback():
    bench = FunctorchFallbackBenchmark(
        op_name="_test_functorch_fallback",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_functorch_fallback,
        gems_op=getattr(flag_gems, "_test_functorch_fallback", None),
        dtypes=_TFF_DTYPES,
        device="cpu",
    )
    bench.run()
