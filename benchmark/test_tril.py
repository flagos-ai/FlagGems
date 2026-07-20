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


def _tril_out_transposed_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out_shape = (*shape[:-2], shape[-1], shape[-2])
    out = torch.empty(out_shape, dtype=dtype, device=device).transpose(-2, -1)
    yield inp, 0, {"out": out}


def _tril_out_sliced_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    if len(shape) == 2:
        out = torch.empty((shape[0] * 2, shape[1]), dtype=dtype, device=device)[::2]
    else:
        out_shape = (shape[0] * 2, *shape[1:])
        out = torch.empty(out_shape, dtype=dtype, device=device)[::2]
    yield inp, 0, {"out": out}


def _tril_extreme_diagonal_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    n = shape[-1]
    for diagonal in (-n, -(n // 2), -1, 0, 1, n // 2, n):
        yield inp, diagonal


def _torch_tril_inplace(inp, diagonal=0):
    """Wrapper for tensor.tril_() method to match torch op signature."""
    return inp.tril_(diagonal)


@pytest.mark.tril
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tril():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tril_extreme_diagonal():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_tril_extreme_diagonal_input_fn,
        op_name="tril",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tril_out_transposed():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_tril_out_transposed_input_fn,
        op_name="tril_out",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril
@pytest.mark.tril_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tril_out_sliced():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_tril_out_sliced_input_fn,
        op_name="tril_out",
        torch_op=torch.tril,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tril_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tril_inplace():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="tril_",
        torch_op=_torch_tril_inplace,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
