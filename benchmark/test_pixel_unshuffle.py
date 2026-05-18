import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, downscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, downscale_factor


class PixelUnshuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((1, 3, 8, 8), 2),
            ((2, 4, 12, 6), 3),
            ((4, 16, 64, 48), 4),
        ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.pixel_unshuffle
def test_pixel_unshuffle():
    bench = PixelUnshuffleBenchmark(
        op_name="pixel_unshuffle",
        torch_op=torch.ops.aten.pixel_unshuffle,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, downscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    N, C, H, W = shape
    r = downscale_factor
    out = torch.empty((N, C * r * r, H // r, W // r), dtype=dtype, device=device)
    yield x, downscale_factor, {"out": out}


class PixelUnshuffleOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((1, 3, 8, 8), 2),
            ((2, 4, 12, 6), 3),
            ((4, 16, 64, 48), 4),
        ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, cur_dtype, self.device)


@pytest.mark.pixel_unshuffle_out
def test_pixel_unshuffle_out():
    bench = PixelUnshuffleOutBenchmark(
        op_name="pixel_unshuffle_out",
        torch_op=torch.ops.aten.pixel_unshuffle.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
