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


def _input_fn_factory(reduce):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        size_dim = shape[dim]
        index = torch.randint(0, size_dim, shape, dtype=torch.long, device=device)
        src = torch.randn(shape, dtype=dtype, device=device)
        yield inp, dim, index, src, {"reduce": reduce}

    return inner


@pytest.mark.scatter_reduce_two_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_sum():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("sum"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_amax():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("amax"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_amin():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("amin"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_two_inplace_mean():
    bench = base.GenericBenchmark2DOnly(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=_input_fn_factory("mean"),
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()
