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

SHAPES = [(1, 64), (32, 64), (128, 256), (1024, 1000), (4096, 1000)]
DTYPES = consts.FLOAT_DTYPES + ([torch.float64] if flag_gems.runtime.device.support_fp64 else [])


class MultiMarginBackwardBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = SHAPES

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SHAPES)
    DEFAULT_SHAPE_DESC = "N, C"


def _case_fn_factory(p, weighted, reduction):
    def case_fn(shape, dtype):
        del dtype
        n, classes = shape
        yield base.BenchmarkCasePlan(
            shape={"input": [n, classes], "target": [n]},
            params={"p": p, "weighted": weighted, "reduction": reduction},
            builder_args=((n, classes), p, weighted, reduction),
        )
    return case_fn


def _build_inputs(plan, dtype, device):
    (n, classes), p, weighted, reduction = plan.builder_args
    if flag_gems.vendor_name == "enflame" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("Enflame native multi_margin_loss backward does not support float16 or bfloat16")
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.bfloat16:
        pytest.skip("Kunlunxin native multi_margin_loss backward does not support bfloat16")
    torch.manual_seed(0)
    inp = torch.randn((n, classes), dtype=dtype, device=device)
    target = torch.randint(0, classes, (n,), dtype=torch.int64, device=device)
    weight = torch.randn((classes,), dtype=dtype, device=device) if weighted else None
    grad_shape = (n,) if reduction == 0 else ()
    grad_output = torch.randn(grad_shape, dtype=dtype, device=device)
    return grad_output, inp, target, p, 1.0, weight, reduction

@pytest.mark.multi_margin_loss_backward
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss_backward(p, weighted, reduction):
    MultiMarginBackwardBenchmark(
        op_name="multi_margin_loss_backward",
        case_fn=_case_fn_factory(p, weighted, reduction),
        build_inputs_fn=_build_inputs,
        torch_op=torch.ops.aten.multi_margin_loss_backward,
        dtypes=DTYPES,
    ).run()
