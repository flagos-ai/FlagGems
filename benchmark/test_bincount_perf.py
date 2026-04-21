import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark


class BincountBenchmark(Benchmark):
    """Benchmark for bincount operation."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1000, 100),
            (10000, 100),
            (10000, 1000),
            (100000, 100),
            (100000, 1000),
            (1000000, 100),
            (1000000, 1000),
            (1000000, 10000),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.bincount_input_fn(config, cur_dtype, self.device)

    def bincount_input_fn(self, config, dtype, device):
        input_size, max_val = config
        inp = torch.randint(0, max_val, (input_size,), dtype=torch.int64, device=device)
        yield inp,


class BincountWeightsBenchmark(Benchmark):
    """Benchmark for bincount operation with weights."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1000, 100),
            (10000, 100),
            (10000, 1000),
            (100000, 100),
            (100000, 1000),
            (1000000, 100),
            (1000000, 1000),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.bincount_weights_input_fn(config, cur_dtype, self.device)

    def bincount_weights_input_fn(self, config, dtype, device):
        input_size, max_val = config
        inp = torch.randint(0, max_val, (input_size,), dtype=torch.int64, device=device)
        weights = torch.randn(input_size, dtype=dtype, device=device)
        yield inp, {"weights": weights}


@pytest.mark.bincount
def test_bincount():
    bench = BincountBenchmark(
        op_name="bincount",
        torch_op=torch.bincount,
        dtypes=[torch.int64],
    )
    bench.run()


@pytest.mark.bincount
def test_bincount_with_weights():
    bench = BincountWeightsBenchmark(
        op_name="bincount_with_weights",
        torch_op=torch.bincount,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
