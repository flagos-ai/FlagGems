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


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES (Issue #2693).
@pytest.mark.div_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_div():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_div_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor_",
        torch_op=lambda a, b: a.div_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _div_tensor_mode_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    if dtype in consts.FLOAT_DTYPES:
        inp2 = torch.where(inp2 >= 0, inp2 + 0.1, inp2 - 0.1)
    else:
        inp2 = torch.where(inp2 == 0, 1, inp2)
    yield inp1, inp2


def _div_scalar_mode_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    scalar = -2.5 if dtype in consts.FLOAT_DTYPES else -3
    yield inp, scalar


def _div_mode_dtypes(rounding_mode):
    return [torch.float32] if rounding_mode == "trunc" else consts.FLOAT_DTYPES


@pytest.mark.div_tensor_mode
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_tensor_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="div_tensor_mode",
        input_fn=_div_tensor_mode_input_fn,
        torch_op=lambda a, b: torch.div(a, b, rounding_mode=rounding_mode),
        dtypes=_div_mode_dtypes(rounding_mode),
    )
    bench.run()


@pytest.mark.div_tensor_mode_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_tensor_mode_inplace(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="div_tensor_mode_",
        input_fn=_div_tensor_mode_input_fn,
        torch_op=lambda a, b: a.div_(b, rounding_mode=rounding_mode),
        dtypes=_div_mode_dtypes(rounding_mode),
        is_inplace=True,
    )
    bench.run()


@pytest.mark.div_scalar_mode
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_scalar_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="div_scalar_mode",
        input_fn=_div_scalar_mode_input_fn,
        torch_op=lambda a, b: torch.div(a, b, rounding_mode=rounding_mode),
        dtypes=_div_mode_dtypes(rounding_mode),
    )
    bench.run()


@pytest.mark.div_scalar_mode_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_scalar_mode_inplace(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="div_scalar_mode_",
        input_fn=_div_scalar_mode_input_fn,
        torch_op=lambda a, b: a.div_(b, rounding_mode=rounding_mode),
        dtypes=_div_mode_dtypes(rounding_mode),
        is_inplace=True,
    )
    bench.run()


@pytest.mark.div_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_div_scalar_inplace():
    def input_fn(shape, dtype, device):
        inp = utils.generate_tensor_input(shape, dtype, device)
        yield inp, 0.001

    bench = base.GenericBenchmark(
        op_name="div_scalar_",
        input_fn=input_fn,
        torch_op=lambda a, b: a.div_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.div_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_div_out():
    def input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        out = torch.empty_like(inp1)
        yield inp1, inp2, {"out": out}

    bench = base.GenericBenchmark(
        op_name="div_out",
        input_fn=input_fn,
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
