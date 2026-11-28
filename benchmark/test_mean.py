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

from . import base, consts, utils


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.mean
def test_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="mean", torch_op=torch.mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def _mean_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, [1]


def _gems_mean(inp, dim=None, keepdim=False):
    if dim is None:
        return flag_gems.mean(inp)
    return flag_gems.mean_dim(inp, dim, keepdim)


@pytest.mark.mean_backward
def test_mean_backward():
    # UnaryReductionBenchmark emits one case per default shape: the 1-D shape
    # exercises the all-reduce ``mean`` overload, every other shape uses
    # ``mean.dim`` with dim=1.  Both overloads are covered by this one mark.
    bench = base.UnaryReductionBenchmark(
        op_name="mean",
        torch_op=torch.mean,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(_gems_mean)
    bench.run()


@pytest.mark.mean_dim_backward
def test_mean_dim_backward():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_mean_dim_input_fn,
        op_name="mean_dim",
        torch_op=torch.mean,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(_gems_mean)
    bench.run()


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.mean_dim
def test_mean_dim():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_mean_dim_input_fn,
        op_name="mean_dim",
        torch_op=torch.mean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
