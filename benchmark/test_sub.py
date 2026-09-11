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


@pytest.mark.sub
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sub():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


# TODO(Qiming): Check why we don't have complex type here
@pytest.mark.sub_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_sub_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub_",
        torch_op=lambda a, b: a.sub_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
