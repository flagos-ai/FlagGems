import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReflectionPad1dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33), (1, 1)),
            ((2, 4, 64), (3, 5)),
            ((8, 16, 256), (8, 8)),
            ((32, 64, 2048), (3, 5)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.reflection_pad1d
def test_reflection_pad1d():
    bench = ReflectionPad1dBenchmark(
        op_name="reflection_pad1d",
        torch_op=torch.ops.aten.reflection_pad1d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _input_fn_out(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    out_shape = list(shape)
    out_shape[-1] = out_shape[-1] + padding[0] + padding[1]
    out = torch.empty(tuple(out_shape), dtype=dtype, device=device)
    yield x, list(padding), {"out": out}


class ReflectionPad1dOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((3, 33), (1, 1)),
            ((2, 4, 64), (3, 5)),
            ((8, 16, 256), (8, 8)),
            ((32, 64, 2048), (3, 5)),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn_out(config, dtype, self.device)


@pytest.mark.reflection_pad1d_out
def test_reflection_pad1d_out():
    bench = ReflectionPad1dOutBenchmark(
        op_name="reflection_pad1d_out",
        torch_op=torch.ops.aten.reflection_pad1d.out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
