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

"""Benchmark for ``record_stream``.

Two-phase benchmark: ``_case_fn`` alone produces the case list (no tensor and no
stream exists for ``--list-cases``) and ``_build_inputs_fn`` materializes the
tensor plus the stream for exactly those plans at execution time.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

_OP_NAME = "record_stream"

# Recording is pure storage bookkeeping, so a shape only scales how much storage
# the allocator must keep alive while the recorded stream is pending.
_BENCH_SHAPES = [
    (256,),
    (1024,),
    (1024, 1024),
    (20, 320, 15),
]

# A fresh stream is the cross-stream hand-off; the current stream is the
# training-loop pattern, where the tensor is recorded on the stream already in use.
_STREAM_KINDS = ["fresh", "current"]


def _case_fn(shape, dtype):
    del dtype
    for stream_kind in _STREAM_KINDS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"stream": stream_kind},
            builder_args=(shape, stream_kind),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, stream_kind = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    stream = (
        torch.accelerator.current_stream()
        if stream_kind == "current"
        else torch.Stream()
    )
    return inp, stream


class RecordStreamBenchmark(OperatorBenchmark):
    """Forwards the stream operand unchanged to reference and candidate."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        for shape in _BENCH_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)

    def unpack_to_args_kwargs(self, input):
        # The default unpacking keeps tensors and primitives only, which would
        # drop the stream and change the call for both sides.
        inp, stream = input
        return (inp, stream), {}


@pytest.mark.record_stream
def test_record_stream():
    bench = RecordStreamBenchmark(
        op_name=_OP_NAME,
        torch_op=torch.ops.aten.record_stream,
        gems_op=getattr(flag_gems, _OP_NAME, None),
        dtypes=consts.FLOAT_DTYPES,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
    )
    bench.run()
