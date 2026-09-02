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


def unique_dim_consecutive_input_fn(shape, dtype, device):
    # Generate input tensor for unique_dim_consecutive
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": 0, "return_inverse": True, "return_counts": True}


def unique_dim_consecutive_out_input_fn(shape, dtype, device):
    # Generate input tensor and out tensors for unique_dim_consecutive.out
    inp = utils.generate_tensor_input(shape, dtype, device)
    out0 = torch.empty(0, dtype=dtype, device=device)
    out1 = torch.empty(0, dtype=torch.int64, device=device)
    out2 = torch.empty(0, dtype=torch.int64, device=device)
    yield inp, {
        "dim": 0,
        "return_inverse": True,
        "return_counts": True,
        "out0": out0,
        "out1": out1,
        "out2": out2,
    }


@pytest.mark.unique_dim_consecutive
def test_unique_dim_consecutive():
    bench = base.GenericBenchmark(
        op_name="unique_dim_consecutive",
        torch_op=torch.unique_consecutive,
        gems_op=flag_gems.unique_dim_consecutive,
        input_fn=unique_dim_consecutive_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unique_dim_consecutive_out
def test_unique_dim_consecutive_out():
    bench = base.GenericBenchmark(
        op_name="unique_dim_consecutive_out",
        torch_op=torch.ops.aten.unique_consecutive.out,
        gems_op=flag_gems.unique_dim_consecutive_out,
        input_fn=unique_dim_consecutive_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
