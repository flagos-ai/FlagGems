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

CASES = [(64,16,64,64,3,1),(128,32,128,128,3,1),(256,16,256,256,5,2),(512,8,128,128,3,1),(1024,4,64,64,7,3)]

class ConvTbcBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "time, batch, Cin, Cout, kernel, pad"

def _case_fn(shape, dtype):
    del dtype
    case = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"case": list(case)}, params={}, builder_args=(case,))

def _build_inputs_fn(plan, dtype, device):
    ilen,batch,cin,cout,kw,pad = plan.builder_args[0]
    olen = ilen - kw + 1 + 2 * pad
    torch.manual_seed(0)
    return (torch.randn((olen,batch,cout),dtype=dtype,device=device), torch.randn((ilen,batch,cin),dtype=dtype,device=device), torch.randn((kw,cin,cout),dtype=dtype,device=device), torch.randn((cout,),dtype=dtype,device=device), pad)

@pytest.mark.conv_tbc_backward
def test_conv_tbc_backward():
    ConvTbcBackwardBenchmark(op_name="conv_tbc_backward", torch_op=torch.ops.aten.conv_tbc_backward.default, case_fn=_case_fn, build_inputs_fn=_build_inputs_fn, dtypes=consts.FLOAT_DTYPES).run()
