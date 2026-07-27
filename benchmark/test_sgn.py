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

import warnings

import pytest
import torch

import flag_gems

from . import base, consts

SGN_REAL_DTYPES = (
    consts.FLOAT_DTYPES
    + ([torch.float64] if flag_gems.runtime.device.support_fp64 else [])
    + [torch.int8, torch.uint8]
    + consts.INT_DTYPES
    + ([torch.int64] if flag_gems.runtime.device.support_int64 else [])
    + consts.BOOL_DTYPES
)
SGN_COMPLEX_DTYPES = [torch.complex32] + consts.COMPLEX_DTYPES
SGN_DTYPES = SGN_REAL_DTYPES + SGN_COMPLEX_DTYPES


def _make_input(shape, dtype, device):
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int8, device=device).bool()
    if dtype is torch.uint8:
        return torch.randint(0, 100, shape, dtype=dtype, device=device)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        return torch.randint(-100, 100, shape, dtype=dtype, device=device)
    if dtype is torch.complex32:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="ComplexHalf support is experimental.*",
                category=UserWarning,
            )
            return torch.randn(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


class SgnBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield (_make_input(shape, dtype, self.device),)


class SgnOutBenchmark(base.UnaryPointwiseOutBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = _make_input(shape, dtype, self.device)
            yield inp, {"out": torch.empty_like(inp)}


@pytest.mark.sgn
def test_sgn():
    bench = SgnBenchmark(
        op_name="sgn",
        torch_op=torch.sgn,
        dtypes=SGN_DTYPES,
    )
    bench.run()


@pytest.mark.sgn_
def test_sgn_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sgn_",
        torch_op=lambda a: a.sgn_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.sgn_out
def test_sgn_out():
    bench = SgnOutBenchmark(
        op_name="sgn_out",
        torch_op=torch.sgn,
        dtypes=SGN_DTYPES,
    )
    bench.run()
