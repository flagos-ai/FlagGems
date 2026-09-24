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

import math

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts

# ``_neg_view_copy`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register the markers
# directly on the MarkGenerator so ``@pytest.mark._neg_view_copy`` and
# ``-m _neg_view_copy`` both work.
setattr(
    pytest.mark,
    "_neg_view_copy",
    MarkDecorator(Mark("_neg_view_copy", (), {}, _ispytest=True), _ispytest=True),
)
setattr(
    pytest.mark,
    "_neg_view_copy_out",
    MarkDecorator(Mark("_neg_view_copy_out", (), {}, _ispytest=True), _ispytest=True),
)

# _neg_view_copy materializes a fresh copy (negated values) of the input, so
# each case needs input + output (2x one tensor's memory per case). The default
# shape set contains 1G-element core shapes whose fp32 input+output would need
# ~8 GiB and OOM on busy GPUs; cap the shapes while keeping
# performance-relevant sizes (2**26 elements = 256 MiB fp32 per tensor).
MAX_ELEMENTS = 2**26


class NegViewCopyBenchmark(base.UnaryPointwiseBenchmark):
    # The op is a pure unary pointwise copy, so the public unary pointwise
    # family covers its call semantics (single input tensor -> fresh output).
    MAX_ELEMENTS = MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


class NegViewCopyOutBenchmark(base.UnaryPointwiseOutBenchmark):
    # The .out overload writes into a caller-supplied buffer; the public unary
    # pointwise-out family builds that buffer and passes it as ``out=``.
    MAX_ELEMENTS = MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


@pytest.mark._neg_view_copy
def test__neg_view_copy():
    bench = NegViewCopyBenchmark(
        op_name="_neg_view_copy",
        torch_op=torch.ops.aten._neg_view_copy,
        # flag_gems._neg_view_copy is not registered as a direct callable yet;
        # KernelGen's --override _neg_view_copy:<file>:<function> still wins at run
        # time through Benchmark._candidate_call.
        gems_op=getattr(flag_gems, "_neg_view_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark._neg_view_copy_out
def test__neg_view_copy_out():
    bench = NegViewCopyOutBenchmark(
        op_name="_neg_view_copy",
        torch_op=torch.ops.aten._neg_view_copy.out,
        gems_op=getattr(flag_gems, "_neg_view_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
