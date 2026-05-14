import os

import pytest
import torch

import flag_gems

from . import base, consts

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
    DEFAULT_SHAPES = [
        (1048576,),
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (1024, 1024, 1024),
    ]
    DEFAULT_SHAPE_DESC = "data shape"
    is_segment_backward = False
    use_backward_out = False
    use_out = False

    def set_shapes(self, shape_file_path=None):
        default_shape_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
        )
        if shape_file_path and os.path.abspath(shape_file_path) != default_shape_file:
            super().set_shapes(shape_file_path)
            return

        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

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
def test_segment_reduce():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce",
        torch_op=torch.segment_reduce,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.segment_reduce_out
def test_segment_reduce_out():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce_out",
        torch_op=torch.ops.aten.segment_reduce.out,
        dtypes=consts.FLOAT_DTYPES,
        use_out=True,
    )
    bench.run()


@pytest.mark.segment_reduce_backward
def test_segment_reduce_backward():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce_backward",
        torch_op=torch.ops.aten._segment_reduce_backward,
        dtypes=consts.FLOAT_DTYPES,
        is_segment_backward=True,
    )
    bench.run()


@pytest.mark.segment_reduce_backward_out
def test_segment_reduce_backward_out():
    bench = SegmentReduceBenchmark(
        op_name="segment_reduce_backward_out",
        torch_op=torch.ops.aten._segment_reduce_backward.out,
        dtypes=consts.FLOAT_DTYPES,
        use_backward_out=True,
    )
    bench.run()
