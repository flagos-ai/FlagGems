import pytest
import torch

from . import base, consts

PIXEL_SHUFFLE_SHAPES = [
    ((1, 4, 2, 3), 2),
    ((2, 9, 4, 4), 3),
    ((4, 64, 32, 32), 2),
    ((2, 128, 64, 64), 2),
    ((1, 64, 16, 16), 4),
    ((8, 36, 64, 64), 3),
    ((1, 16, 128, 128), 2),
]


def _input_fn(config, dtype, device):
    shape, upscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, upscale_factor


class PixelShuffleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PIXEL_SHUFFLE_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.pixel_shuffle
def test_pixel_shuffle():
    bench = PixelShuffleBenchmark(
        op_name="pixel_shuffle",
        torch_op=torch.ops.aten.pixel_shuffle,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, upscale_factor = config
    x = torch.randn(shape, dtype=dtype, device=device)
    N, C, H, W = shape
    r = upscale_factor
    out = torch.empty((N, C // (r * r), H * r, W * r), dtype=dtype, device=device)
    yield x, upscale_factor, {"out": out}


class PixelShuffleOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PIXEL_SHUFFLE_SHAPES

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, cur_dtype, self.device)


@pytest.mark.pixel_shuffle_out
def test_pixel_shuffle_out():
    bench = PixelShuffleOutBenchmark(
        op_name="pixel_shuffle_out",
        torch_op=torch.ops.aten.pixel_shuffle.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
