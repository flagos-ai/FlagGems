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

"""Benchmark for ``aten::new_zeros``.

``new_zeros`` allocates and zero-fills a fresh buffer; the timed work is the
allocation plus the store stream, and the source tensor only supplies the
default dtype/device metadata. ``torch_op`` is the ATen reference (the perf
baseline) and ``gems_op`` is the FlagGems candidate, both called with the same
``(self, size)`` semantics.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# Fill-stream sizes: the largest is 2**20 elements (4 MiB fp32), which keeps the
# kernel bandwidth-relevant while staying allocation-friendly; the small rows
# measure the launch-bound regime.
NEW_ZEROS_SHAPES = [
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "size": shape},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    size = plan.builder_args[0]
    # The source only supplies defaults; a small tensor keeps the allocation
    # cheap beside the timed zero-fill of `size`.
    src = torch.empty(1, dtype=dtype, device=device)
    return src, list(size)


class NewZerosBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # core_shapes.yaml has no entry for this factory and the base-class
        # fallback includes shapes whose allocation cost dominates the
        # measurement; the curated list above is pinned instead.
        del shape_file_path
        self.shapes = list(NEW_ZEROS_SHAPES)
        self.shape_desc = "size"


@pytest.mark.new_zeros
def test_new_zeros():
    bench = NewZerosBenchmark(
        op_name="new_zeros",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.new_zeros.default,
        gems_op=flag_gems.new_zeros,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _build_inputs_fn_out(plan, dtype, device):
    src, size = _build_inputs_fn(plan, dtype, device)
    # The out buffer is allocated here so its allocation stays outside the
    # timed region; the timed work is the in-place zero fill.
    out = torch.empty(size, dtype=dtype, device=device)
    return src, size, {"out": out}


class NewZerosOutBenchmark(NewZerosBenchmark):
    """out= overload: the caller's buffer is zeroed in place and returned."""


@pytest.mark.new_zeros_out
def test_new_zeros_out():
    bench = NewZerosOutBenchmark(
        op_name="new_zeros_out",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten.new_zeros.out,
        gems_op=flag_gems.new_zeros_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
