import math
import os

import pytest
import torch

from flag_gems.utils import shape_utils

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


def _segment_reduce_op(reduce):
    def inner(data, lengths, axis):
        return torch.segment_reduce(
            data,
            reduce,
            lengths=lengths,
            axis=axis,
            unsafe=True,
        )

    return inner


class SegmentReduceBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(1024,), (512, 128), (64, 128, 128)]
    DEFAULT_SHAPE_DESC = "data shape"
    DEFAULT_SHAPE_FILES = os.path.join(
        os.path.dirname(__file__), "test_segment_reduce.yaml"
    )

    def init_user_config(self):
        super().init_user_config()
        default_shape_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "core_shapes.yaml")
        )
        if os.path.abspath(base.Config.shape_file) == default_shape_file:
            self.set_shapes(self.DEFAULT_SHAPE_FILES)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return [(65536,), (2048, 256), (128, 256, 128)]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            axis = _select_axis(shape)
            data = torch.randn(shape, dtype=cur_dtype, device=self.device)
            lengths = _make_lengths(shape, axis, self.device)
            yield data, lengths, axis

    def get_gbps(self, args, latency=None):
        data, lengths, axis = args
        output_shape = tuple(lengths.shape) + tuple(data.shape[axis + 1 :])
        output = torch.empty(output_shape, dtype=data.dtype, device=data.device)
        io_tensors = [data, lengths, output]
        if self.is_backward:
            io_tensors.extend([output, data])
        io_amount = sum(shape_utils.size_in_bytes(item) for item in io_tensors)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        data, lengths, axis = args
        segment_count = lengths.shape[-1]
        inner_size = math.prod(data.shape[axis + 1 :]) if axis + 1 < data.dim() else 1
        return data.numel() + segment_count * inner_size


@pytest.mark.segment_reduce
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_segment_reduce(reduce):
    bench = SegmentReduceBenchmark(
        op_name=f"segment_reduce_{reduce}",
        torch_op=_segment_reduce_op(reduce),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.segment_reduce_backward
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_segment_reduce_backward(reduce):
    bench = SegmentReduceBenchmark(
        op_name=f"_segment_reduce_backward_{reduce}",
        torch_op=_segment_reduce_op(reduce),
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
