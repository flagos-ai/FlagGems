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

from . import base, consts, utils


@pytest.mark.mean
def test_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="mean", torch_op=torch.mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def _mean_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, [1]


@pytest.mark.mean_dim
def test_mean_dim():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_mean_dim_input_fn,
        op_name="mean_dim",
        torch_op=torch.mean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
