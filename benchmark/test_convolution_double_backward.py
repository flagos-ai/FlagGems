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

from . import base
from .generated_operator_utils import OperatorBenchmark

CASES = [(16,32,64,64,64,3,3,1,1,1,1,False,(0,0)),(16,32,64,64,64,3,3,2,1,1,1,False,(0,0)),(8,32,32,32,32,3,3,1,1,1,2,False,(0,0)),(8,64,16,16,32,3,3,2,1,1,1,True,(1,1))]

class ConvDoubleBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self,shape_file_path=None):
        self.shapes=list(CASES)
        self.shape_desc="N, Cin, H, W, Cout, Kh, Kw, stride, padding, dilation, groups, transposed, output_padding"

def _case_fn(shape,dtype):
    del dtype
    case=tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case":[*case[:-1],list(case[-1])]},params={},builder_args=(case,))

def _build_inputs_fn(plan,dtype,device):
    n,cin,h,w,cout,kh,kw,stride,padding,dilation,groups,transposed,output_padding=plan.builder_args[0]
    torch.manual_seed(0)
    x=torch.randn((n,cin,h,w),dtype=dtype,device=device)
    weight_shape=(cin,cout//groups,kh,kw) if transposed else (cout,cin//groups,kh,kw)
    weight=torch.randn(weight_shape,dtype=dtype,device=device)
    bias=torch.randn((cout,),dtype=dtype,device=device)
    if transposed:
        y=torch.nn.functional.conv_transpose2d(x,weight,bias,stride,padding,output_padding,groups,dilation)
    else:
        y=torch.nn.functional.conv2d(x,weight,bias,stride,padding,dilation,groups)
    gO=torch.randn_like(y)
    return (torch.randn_like(x),torch.randn_like(weight),torch.randn_like(bias),gO,weight,x,[stride,stride],[padding,padding],[dilation,dilation],transposed,list(output_padding),groups,(True,True,True))

@pytest.mark.convolution_double_backward
@pytest.mark.underscore_convolution_double_backward
def test_convolution_double_backward():
    ConvDoubleBackwardBenchmark(op_name="_convolution_double_backward",torch_op=torch.ops.aten._convolution_double_backward.default,case_fn=_case_fn,build_inputs_fn=_build_inputs_fn,dtypes=[torch.float32,torch.float16]).run()
