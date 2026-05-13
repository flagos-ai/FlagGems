import pytest
import torch

from . import base, consts


def _input_fn(config, dtype, device):
    shape, padding = config
    x = torch.randn(shape, dtype=dtype, device=device)
    yield x, list(padding)


class ReplicationPad1dBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 3, 7), (1, 2)),
            ((4, 16, 64), (3, 1)),
            ((8, 32, 256), (1, 2)),
            ((32, 256), (3, 1)),
        ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.replication_pad1d
def test_replication_pad1d():
    bench = ReplicationPad1dBenchmark(
        op_name="replication_pad1d",
        torch_op=torch.ops.aten.replication_pad1d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def _torch_replication_pad1d_out(x, padding):
    pad_left, pad_right = padding[0], padding[1]
    if x.dim() == 3:
        N, C, W_in = x.shape
        out_shape = (N, C, W_in + pad_left + pad_right)
    else:
        C, W_in = x.shape
        out_shape = (C, W_in + pad_left + pad_right)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return torch.ops.aten.replication_pad1d.out(x, padding, out=out)


class ReplicationPad1dOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((2, 3, 7), (1, 2)),
            ((4, 16, 64), (3, 1)),
            ((8, 32, 256), (1, 2)),
            ((32, 256), (3, 1)),
        ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from _input_fn(config, dtype, self.device)


@pytest.mark.replication_pad1d_out
def test_replication_pad1d_out():
    bench = ReplicationPad1dOutBenchmark(
        op_name="replication_pad1d_out",
        torch_op=_torch_replication_pad1d_out,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
