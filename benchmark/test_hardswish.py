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

# bf16 excluded: XPU native reference (torch_xmlir xdnn_pytorch_wrapper/
# hardswish.cpp:30) reports "scalar type of ret: kbfloat16 is unsupported"
# -> [NOT IMPLEMENTED] error code=4 for BOTH hardswish_/hardswish/hardswish.out
# on the reference-side latency_base path (NON_BUG, pre-existing, same family as
# acosh/sinh/cosh/mish). bf16 correctness is covered by tests/test_hardswish.py
# --ref cpu (57P incl. bf16).


@pytest.mark.hardswish_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_hardswish_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish_",
        torch_op=torch.ops.aten.hardswish_,
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.hardswish
def test_hardswish():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish",
        torch_op=torch.nn.functional.hardswish,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.hardswish_out
def test_hardswish_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="hardswish_out",
        torch_op=torch.ops.aten.hardswish.out,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
