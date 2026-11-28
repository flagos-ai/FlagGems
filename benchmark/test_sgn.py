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


class SgnBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def unpack_to_args_kwargs(self, input_tuple):
        args, kwargs = super().unpack_to_args_kwargs(input_tuple)
        args = [
            (
                arg.requires_grad_(True)
                if torch.is_tensor(arg) and arg.is_complex()
                else arg
            )
            for arg in args
        ]
        return args, kwargs


@pytest.mark.sgn
def test_sgn():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sgn",
        torch_op=torch.sgn,
        dtypes=consts.FLOAT_DTYPES,
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
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="sgn_out",
        torch_op=torch.sgn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.sgn_backward
def test_sgn_backward():
    bench = SgnBackwardBenchmark(
        op_name="sgn_backward",
        torch_op=torch.sgn,
        gems_op=flag_gems.sgn,
        # Real sgn has a structurally zero derivative; benchmark the non-trivial
        # complex derivative that the fused Triton kernel computes.
        dtypes=consts.COMPLEX_DTYPES,
        is_backward=True,
    )
    bench.run()
