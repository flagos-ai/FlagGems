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

"""Benchmark for ``aten::empty_like`` and its native ``.out`` overload.

The operator only allocates, so latency/speedup are the meaningful metrics and
no measurement reads the uninitialized result. Case planning creates no tensors,
which keeps ``--list-cases`` allocation-free and identical to what ``--case-id``
replay executes.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.BOOL_DTYPES
    + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
)

MEMORY_FORMATS = {
    "preserve_format": torch.preserve_format,
    "contiguous_format": torch.contiguous_format,
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
}

# channels_last / channels_last_3d are only accepted for rank-4 / rank-5 inputs.
_CHANNELS_LAST_RANKS = (("channels_last", 4), ("channels_last_3d", 5))

# The shared default shapes are all rank 1-3, so the rank-4 / rank-5 geometry that
# the two channels-last formats require is appended on top of them (and on top of
# any caller-supplied shape file) instead of replacing the shared workloads.
_RANK_BOUNDARY_SHAPES = [(16, 128, 64, 60), (16, 7, 57, 32, 29)]


def _case_fn(shape, dtype):
    del dtype
    names = ["preserve_format", "contiguous_format"]
    names += [name for name, rank in _CHANNELS_LAST_RANKS if len(shape) == rank]
    for name in names:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"memory_format": name},
            builder_args=(shape, name),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, name = plan.builder_args
    # Only the layout matters: the result of empty_like is never read, so an
    # uninitialized allocation is enough and the same memory_format reaches the
    # reference and the candidate.
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, {"memory_format": MEMORY_FORMATS[name]}


def _out_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"out": True},
        builder_args=(shape,),
    )


def _build_out_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, {"out": torch.empty(shape, dtype=dtype, device=device)}


class EmptyLikeBenchmark(OperatorBenchmark):
    """Shared default shapes plus the rank-4 / rank-5 channels-last geometry."""

    def set_shapes(self, shape_file_path=None):
        # Keeps the operator/class entry of any caller-supplied shape file (and
        # the shared defaults, including the COMPREHENSIVE extras merged by the
        # base class) while guaranteeing the ranks the format cases need.
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape) for shape in list(self.shapes) + _RANK_BOUNDARY_SHAPES
            )
        )


# ``flag_gems.empty_like`` is not implemented in every checkout;
# ``getattr(..., None)`` keeps listing and import working, while execution still
# fails without an injected candidate.
@pytest.mark.empty_like
def test_empty_like():
    bench = EmptyLikeBenchmark(
        op_name="empty_like",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.empty_like,
        gems_op=getattr(flag_gems, "empty_like", None),
        dtypes=DTYPES,
    )
    bench.run()


@pytest.mark.empty_like
def test_empty_like_out():
    # The reference is the real native ``.out`` overload, called with the same
    # (input, out=buffer) semantics as the candidate.
    bench = EmptyLikeBenchmark(
        op_name="empty_like",
        case_fn=_out_case_fn,
        build_inputs_fn=_build_out_inputs_fn,
        torch_op=torch.ops.aten.empty_like.out,
        gems_op=getattr(flag_gems, "empty_like", None),
        dtypes=DTYPES,
    )
    bench.run()
