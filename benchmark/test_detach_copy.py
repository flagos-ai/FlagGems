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

import flag_gems

from . import base, consts

# detach_copy is a pure memory copy: unary, pointwise in shape space and
# dtype-agnostic. The public UnaryPointwiseBenchmark family covers the .default
# overload (single tensor in, fresh tensor out) and UnaryPointwiseOutBenchmark
# covers the .out overload (caller-supplied out buffer passed as a kwarg). Both
# use the same call semantics as torch.ops.aten.detach_copy / .out, which are
# the perf comparison reference; the candidate is passed explicitly via gems_op
# (the harness override wins through Benchmark._candidate_call inside the Benchmark base).
#
# flag_gems exposes no direct detach_copy callable today, so gems_op falls back
# to None and the KernelGen override registry provides the candidate; the YAML
# shape set contains 1e9-element tensors (~4 GiB fp32 per tensor), so the
# MAX_ELEMENTS cap keeps every timed copy within a few hundred MB.
_MAX_ELEMENTS = 2**26


class DetachCopyBenchmark(base.UnaryPointwiseBenchmark):
    MAX_ELEMENTS = _MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


class DetachCopyOutBenchmark(base.UnaryPointwiseOutBenchmark):
    MAX_ELEMENTS = _MAX_ELEMENTS

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]


@pytest.mark.detach_copy
def test_detach_copy():
    bench = DetachCopyBenchmark(
        op_name="detach_copy",
        torch_op=torch.ops.aten.detach_copy,
        gems_op=getattr(flag_gems, "detach_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.detach_copy_out
def test_detach_copy_out():
    bench = DetachCopyOutBenchmark(
        op_name="detach_copy",
        torch_op=torch.ops.aten.detach_copy.out,
        gems_op=getattr(flag_gems, "detach_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
