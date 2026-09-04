import pytest
import torch

from . import base, consts

DEFAULT_SHAPE_DESC = "M, N, K1, K2"


class TensordotBenchmark(base.Benchmark):
    """Benchmark for tensordot: contract the last two dims of a (dims=2)."""

    DEFAULT_SHAPE_DESC = "M, N, K1, K2"

    def set_shapes(self, shape_file_path=None):
        # (M, N, K1, K2): a is (M, K1, K2), b is (K1, K2, N); contract dims=2.
        self.shapes = [
            (64, 64, 16, 16),
            (128, 128, 32, 32),
            (256, 256, 32, 32),
            (512, 512, 32, 32),
            (1024, 512, 64, 16),
            (1024, 1024, 64, 16),
        ]

    def get_input_iter(self, dtype):
        for m, n, k1, k2 in self.shapes:
            a = torch.randn((m, k1, k2), dtype=dtype, device=self.device)
            b = torch.randn((k1, k2, n), dtype=dtype, device=self.device)
            yield a, b, 2


class TensordotOutBenchmark(TensordotBenchmark):
    def get_input_iter(self, dtype):
        for m, n, k1, k2 in self.shapes:
            yield from _input_fn_out(m, n, k1, k2, dtype, self.device)


def _input_fn_out(m, n, k1, k2, dtype, device):
    a = torch.randn((m, k1, k2), dtype=dtype, device=device)
    b = torch.randn((k1, k2, n), dtype=dtype, device=device)
    out = torch.empty((m, n), dtype=dtype, device=device)
    yield a, b, 2, {"out": out}


@pytest.mark.tensordot
def test_tensordot():
    bench = TensordotBenchmark(
        op_name="tensordot",
        torch_op=torch.tensordot,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.tensordot_out
def test_tensordot_out():
    bench = TensordotOutBenchmark(
        op_name="tensordot_out",
        torch_op=torch.tensordot,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
