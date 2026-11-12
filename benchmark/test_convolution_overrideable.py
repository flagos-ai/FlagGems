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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CASES = [(16,32,24,24,24,3,3,1,1,1,1),(32,64,64,64,32,3,3,1,1,1,1),(32,64,128,128,32,3,3,1,1,1,1),(16,32,24,24,24,3,3,2,1,1,1),(32,64,64,64,32,5,5,1,2,1,1)]

class ConvolutionOverrideableBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "N, Cin, H, W, Cout, Kh, Kw, stride, padding, dilation, groups"

def _case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case": list(case)}, params={}, builder_args=(case,))

def _base_inputs(case, dtype, device):
    n,cin,h,w,cout,kh,kw,stride,padding,dilation,groups = case
    torch.manual_seed(0)
    inp = torch.randn((n,cin,h,w),dtype=dtype,device=device)
    weight = torch.randn((cout,cin//groups,kh,kw),dtype=dtype,device=device)
    args = (inp,weight,None,[stride,stride],[padding,padding],[dilation,dilation],False,[0,0],groups)
    return args, (n,cout,(h+2*padding-dilation*(kh-1)-1)//stride+1,(w+2*padding-dilation*(kw-1)-1)//stride+1)

def _build_inputs_fn(plan, dtype, device):
    (case,) = plan.builder_args
    args, _ = _base_inputs(case,dtype,device)
    return args

@pytest.mark.convolution_overrideable
def test_convolution_overrideable():
    ConvolutionOverrideableBenchmark(op_name="convolution_overrideable", torch_op=torch.ops.aten.convolution.default, gems_op=flag_gems.convolution_overrideable, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()


def _build_out_inputs(plan, dtype, device):
    (case,) = plan.builder_args
    args, out_shape = _base_inputs(case, dtype, device)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    return (*args, {"out": out})


@pytest.mark.convolution_overrideable_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issue #4131: conv kernels not working",
)
def test_convolution_overrideable_out(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")
    torch.backends.cudnn.allow_tf32 = False

    def stock_out(*args, **kwargs):
        return flag_gems.convolution_overrideable_out(*args, **kwargs)

    bench = ConvolutionOverrideableBenchmark(
        op_name="convolution_overrideable_out",
        torch_op=torch.ops.aten.convolution.out,
        case_fn=_case_fn,
        build_inputs_fn=_build_out_inputs,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.set_gems(stock_out)
    bench.run()
