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


@pytest.mark.bitwise_and_tensor
def test_bitwise_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor",
        torch_op=torch.bitwise_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_and_tensor_
def test_bitwise_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0x3F


@pytest.mark.bitwise_and_scalar
def test_bitwise_and_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="bitwise_and_scalar",
        torch_op=torch.bitwise_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


def bitwise_and_scalar_input_fn(shape, cur_dtype, device):
    inp1 = base.generate_tensor_input(shape, cur_dtype, device)
    if cur_dtype == torch.bool:
        inp2 = True
    else:
        inp2 = 0x00FF
    yield inp1, inp2


@pytest.mark.bitwise_and_scalar_
def test_bitwise_and_scalar_():
    bench = base.GenericBenchmark(
        op_name="bitwise_and_scalar_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        input_fn=bitwise_and_scalar_input_fn,
        is_inplace=True,
    )
    bench.run()


# NOTE (kunlunxin/XPU): the native torch `bitwise_and.Scalar_Tensor` overload is
# broken on XPU (device check error for `torch.bitwise_and(py_int, xpu_tensor)`),
# so the reference side feeds a 0-D tensor of the same dtype/device (dispatches
# to the working `bitwise_and.Tensor`), while the gems side explicitly calls
# `flag_gems.bitwise_and_scalar_tensor` (Scalar_Tensor API). See
# harness/solution/performance/analysis/bitwise_and_scalar_tensor_benchmark_fix.md.
def scalar_tensor_input_fn(shape, cur_dtype, device):
    scalar = 0x96 if cur_dtype != torch.bool else True
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    yield torch.tensor(scalar, dtype=cur_dtype, device=device), inp


@pytest.mark.bitwise_and_scalar_tensor
def test_bitwise_and_scalar_tensor():
    bench = base.GenericBenchmark(
        op_name="bitwise_and_scalar_tensor",
        torch_op=torch.bitwise_and,
        gems_op=lambda a, b: flag_gems.bitwise_and_scalar_tensor(
            bool(a) if a.dtype == torch.bool else int(a), b
        ),
        input_fn=scalar_tensor_input_fn,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
