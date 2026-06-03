import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad2dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33, 33), (1, 1, 1, 1)),
            ((2, 4, 32, 64), (2, 3, 2, 3)),
            ((8, 16, 64, 64), (3, 5, 3, 5)),
            ((32, 64, 128, 256), (0, 4, 0, 4)),
            ((16, 32, 64, 128), (1, 1, 1, 1)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad2d
def test_reflection_pad2d():
    bench = ReflectionPad2dBenchmark(
        op_name="reflection_pad2d",
        torch_op=torch.ops.aten.reflection_pad2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    pad_left, pad_right, pad_top, pad_bottom = padding
    H_out = x.shape[-2] + pad_top + pad_bottom
    W_out = x.shape[-1] + pad_left + pad_right
    out_shape = (*x.shape[:-2], H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    yield x, list(padding), {"out": out}


class ReflectionPad2dOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33, 33), (1, 1, 1, 1)),
            ((2, 4, 32, 64), (2, 3, 2, 3)),
            ((8, 16, 64, 64), (3, 5, 3, 5)),
            ((32, 64, 128, 256), (0, 4, 0, 4)),
            ((16, 32, 64, 128), (1, 1, 1, 1)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, dtype, self.device)


@pytest.mark.reflection_pad2d_out
def test_reflection_pad2d_out():
    bench = ReflectionPad2dOutBenchmark(
        op_name="reflection_pad2d_out",
        torch_op=torch.ops.aten.reflection_pad2d.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _backward_input_fn(config, dtype, device):
    shape, padding = config
    pad_left, pad_right, pad_top, pad_bottom = padding
    H_out = shape[-2] + pad_top + pad_bottom
    W_out = shape[-1] + pad_left + pad_right
    grad_output = torch.randn(
        (*shape[:-2], H_out, W_out), dtype=dtype, device=device
    )
    yield grad_output, shape, list(padding)


class ReflectionPad2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33, 33), (1, 1, 1, 1)),
            ((2, 4, 32, 64), (2, 3, 2, 3)),
            ((8, 16, 64, 64), (3, 5, 3, 5)),
            ((32, 64, 128, 256), (0, 4, 0, 4)),
            ((16, 32, 64, 128), (1, 1, 1, 1)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _backward_input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad2d_backward
def test_reflection_pad2d_backward():
    bench = ReflectionPad2dBackwardBenchmark(
        op_name="reflection_pad2d_backward",
        torch_op=torch.ops.aten.reflection_pad2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
