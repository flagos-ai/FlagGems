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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ["mthreads"],
    reason="Issue #4114: Not supported on Moore Threads MUSA",
)

REDUCTIONS = ("sum", "mean", "max", "min", "prod")


def _select_axis(shape):
    return 0 if len(shape) == 1 else 1


def _make_lengths(shape, axis, device):
    size_axis = shape[axis]
    segment_count = min(64, size_axis)
    base_length = size_axis // segment_count
    remainder = size_axis % segment_count
    lengths = torch.full((segment_count,), base_length, dtype=torch.int64)
    if remainder:
        lengths[:remainder] += 1
    outer_shape = shape[:axis]
    if outer_shape:
        lengths = lengths.expand(*outer_shape, segment_count).clone()
    return lengths.to(device)


class SegmentReduceBenchmark(base.Benchmark):
    is_segment_backward = False
    use_backward_out = False
    use_out = False

    def set_more_shapes(self):
        return [(65536,), (2048, 256), (128, 256, 128)]

    def get_input_iter(self, cur_dtype):
        for reduce in REDUCTIONS:
            for shape in self.shapes:
                axis = _select_axis(shape)
                data = torch.randn(shape, dtype=cur_dtype, device=self.device)
                lengths = _make_lengths(shape, axis, self.device)
                kwargs = {
                    "lengths": lengths,
                    "axis": axis,
                    "unsafe": True,
                }
                if self.is_segment_backward or self.use_backward_out:
                    with flag_gems.use_gems():
                        output = torch.segment_reduce(data, reduce, **kwargs)
                    grad = torch.randn_like(output)
                    backward_kwargs = {
                        "lengths": lengths,
                        "axis": axis,
                    }
                    if self.use_backward_out:
                        backward_kwargs["out"] = torch.empty_like(data)
                    yield grad, output, data, reduce, backward_kwargs
                    continue
                if self.use_out:
                    output_shape = tuple(lengths.shape) + tuple(data.shape[axis + 1 :])
                    kwargs["out"] = torch.empty(
                        output_shape, dtype=cur_dtype, device=self.device
                    )
                yield data, reduce, kwargs

    def get_tflops(self, op, *args, **kwargs):
        if self.is_segment_backward or self.use_backward_out:
            data, lengths, axis = args[2], kwargs["lengths"], kwargs["axis"]
        else:
            data, lengths, axis = args[0], kwargs["lengths"], kwargs["axis"]
        segment_count = lengths.shape[-1]
        inner_size = (
            torch.Size(data.shape[axis + 1 :]).numel() if axis + 1 < data.dim() else 1
        )
        return data.numel() + segment_count * inner_size


@pytest.mark.segment_reduce
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_segment_reduce():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce",
        torch_op=torch.segment_reduce,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.segment_reduce_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_segment_reduce_out():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce_out",
        torch_op=torch.ops.aten.segment_reduce.out,
        dtypes=consts.FLOAT_DTYPES,
        use_out=True,
    )
    bench.run()


@pytest.mark.segment_reduce_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_segment_reduce_backward():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce_backward",
        torch_op=torch.ops.aten._segment_reduce_backward,
        dtypes=consts.FLOAT_DTYPES,
        is_segment_backward=True,
    )
    bench.run()


@pytest.mark.segment_reduce_backward_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_segment_reduce_backward_out():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce_backward_out",
        torch_op=torch.ops.aten._segment_reduce_backward.out,
        dtypes=consts.FLOAT_DTYPES,
        use_backward_out=True,
    )
    bench.run()
