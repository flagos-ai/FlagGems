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


@pytest.mark.add
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_add():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.add_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_add_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add_",
        torch_op=lambda a, b: a.add_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


class AddBackwardBenchmark(base.BinaryPointwiseBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            other = torch.randn(shape, dtype=dtype, device=self.device)
            # A non-unit alpha makes both native and FlagGems execute the scaled
            # second-input gradient instead of benchmarking an identity-only path.
            yield inp, other, {"alpha": 0.5}


@pytest.mark.add_backward
def test_add_backward():
    bench = AddBackwardBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    # The Autograd fallthrough routes torch.add through AddFunction, so the
    # candidate measures the same registered path used by applications.
    bench.run()
