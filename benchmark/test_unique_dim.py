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


def _input_fn_dim0(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": 0,
    },


def _input_fn_dim1(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": 1,
    },


@pytest.mark.unique_dim
def test_unique_dim_dim0():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn_dim0,
        op_name="unique_dim",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.unique_dim
def test_unique_dim_dim1():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn_dim1,
        op_name="unique_dim",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
