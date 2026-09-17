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


@pytest.mark.any
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_any():
    bench = base.UnaryReductionBenchmark(
        op_name="any",
        torch_op=torch.any,
        dtypes=consts.FLOAT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.any_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_any_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="any_dim",
        torch_op=torch.any,
        dtypes=consts.FLOAT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


def any_dims_input_fn(shape, dtype, device):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=dtype, device=device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"dim": [0, 1]}


@pytest.mark.any_dims
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_any_dims():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="any_dims",
        torch_op=torch.any,
        input_fn=any_dims_input_fn,
        dtypes=consts.FLOAT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
