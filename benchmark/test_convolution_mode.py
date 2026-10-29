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

CASES = [(16,32,24,24,24,3,3,1,"valid",1,1),(32,64,128,128,32,3,3,1,"valid",1,1),(32,64,210,210,16,5,5,1,"valid",1,1),(16,32,24,24,24,3,3,1,"same",1,1),(32,64,128,128,32,3,3,1,"same",1,1),(32,64,210,210,16,5,5,1,"same",2,1)]

class ConvolutionModeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "N, Cin, H, W, Cout, Kh, Kw, stride, padding, dilation, groups"

def _case_fn(shape, dtype):
    del dtype
    case=tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case":list(case)},params={},builder_args=(case,))

def _build_inputs_fn(plan,dtype,device):
    n,cin,h,w,cout,kh,kw,stride,padding,dilation,groups=plan.builder_args[0]
    torch.manual_seed(0)
    return (torch.randn((n,cin,h,w),dtype=dtype,device=device),torch.randn((cout,cin//groups,kh,kw),dtype=dtype,device=device),None,(stride,stride),padding,(dilation,dilation),groups)

@pytest.mark.convolution_mode
@pytest.mark.underscore_convolution_mode
def test_convolution_mode():
    ConvolutionModeBenchmark(op_name="_convolution_mode",torch_op=torch.ops.aten._convolution_mode.default,case_fn=_case_fn,build_inputs_fn=_build_inputs_fn,dtypes=consts.FLOAT_DTYPES).run()
