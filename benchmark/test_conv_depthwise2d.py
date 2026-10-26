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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CASES = [(32,4,8,8,2,2,1,0,1),(18,16,4,4,2,2,2,1,1),(1,64,224,224,3,3,1,1,1),(1,128,112,112,5,5,2,2,1)]

class ConvDepthwise2dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "N, C, H, W, Kh, Kw, stride, padding, dilation"

def _case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case": list(case)}, params={}, builder_args=(case,))

def _build_inputs_fn(plan, dtype, device):
    n, c, h, w, kh, kw, stride, padding, dilation = plan.builder_args[0]
    torch.manual_seed(0)
    inp = torch.randn((n,c,h,w), dtype=dtype, device=device)
    weight = torch.randn((c,1,kh,kw), dtype=dtype, device=device)
    return inp, weight, [kh,kw], None, [stride,stride], [padding,padding], [dilation,dilation]

@pytest.mark.conv_depthwise2d
@pytest.mark.underscore_conv_depthwise2d
def test_conv_depthwise2d():
    ConvDepthwise2dBenchmark(op_name="_conv_depthwise2d", torch_op=torch.ops.aten._conv_depthwise2d.default, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
