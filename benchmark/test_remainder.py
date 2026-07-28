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


@pytest.mark.remainder
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_remainder():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder",
        torch_op=torch.remainder,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.remainder_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_remainder_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder_tensor",
        torch_op=torch.remainder,
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.remainder_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_remainder_tensor_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="remainder_tensor_",
        torch_op=lambda a, b: a.remainder_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def remainder_scalar_input_fn(shape, dtype, device):
    inp = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device=device,
    )
    scalar = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        (1,),
        dtype=dtype,
        device=device,
    ).item()
    if scalar == 0:
        scalar = 1
    yield inp, scalar


@pytest.mark.remainder_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_remainder_scalar():
    bench = base.GenericBenchmark(
        input_fn=remainder_scalar_input_fn,
        op_name="remainder_scalar",
        torch_op=torch.remainder,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.remainder_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_remainder_scalar_inplace():
    bench = base.GenericBenchmark(
        input_fn=remainder_scalar_input_fn,
        op_name="remainder_scalar_",
        torch_op=lambda a, b: a.remainder_(b),
        dtypes=consts.INT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def scalar_tensor_remainder_input_fn(shape, dtype, device):
    inp = torch.randint(1, 100, shape, dtype=dtype, device=device)
    scalar = 7
    yield scalar, inp


@pytest.mark.remainder_scalar_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_remainder_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="remainder_scalar_tensor",
        torch_op=torch.remainder,
        input_fn=scalar_tensor_remainder_input_fn,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
