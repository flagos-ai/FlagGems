# Copyright 2026, The FlagOS Contributors.
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

from . import base, consts

# _neg_view_copy materializes a fresh copy (negated values) of the input, so
# each case needs input + output (2x one tensor's memory per case). The default
# shape set contains 1G-element core shapes whose fp32 input+output would need
# ~8 GiB and OOM on busy GPUs; cap the shapes while keeping
# performance-relevant sizes (2**26 elements = 256 MiB fp32 per tensor),
# mirroring the KernelGen acceptance benchmark.
MAX_ELEMENTS = 2**26


class NegViewCopyBenchmark(base.UnaryPointwiseBenchmark):
    """``aten::_neg_view_copy`` over the public unary pointwise shape family.

    The op is a pure unary pointwise copy, so the public unary pointwise
    family covers its call semantics (single input tensor -> fresh output).
    """

    MAX_ELEMENTS = MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Cap shapes to MAX_ELEMENTS: the kernelgen acceptance benchmark limits the
        # core set to these sizes so fp32 input+output fits on a busy H20.
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


class NegViewCopyOutBenchmark(base.UnaryPointwiseOutBenchmark):
    """``.out`` overload writes into a caller-supplied buffer."""

    MAX_ELEMENTS = MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Cap shapes to MAX_ELEMENTS: the kernelgen acceptance benchmark limits the
        # core set to these sizes so fp32 input+output fits on a busy H20.
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


@pytest.mark.neg_view_copy
def test_neg_view_copy():
    bench = NegViewCopyBenchmark(
        op_name="neg_view_copy",
        torch_op=torch.ops.aten._neg_view_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.neg_view_copy_out
def test_neg_view_copy_out():
    bench = NegViewCopyOutBenchmark(
        op_name="neg_view_copy_out",
        torch_op=torch.ops.aten._neg_view_copy.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
