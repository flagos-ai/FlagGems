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


@pytest.mark.logit
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit",
        torch_op=lambda a: torch.logit(a, eps=1e-6),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.logit_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logit_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="logit_",
        torch_op=lambda a: a.logit_(eps=1e-6),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.logit_out
def test_logit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="logit_out",
        torch_op=lambda a, out: torch.logit(a, eps=1e-6, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
