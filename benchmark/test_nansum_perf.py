from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import BenchLevel, FLOAT_DTYPES
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark, generate_tensor_input
from flag_gems.utils import shape_utils

def _inject_nans_(t: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    if not t.is_floating_point():
        return t
    if t.numel() == 0:
        return t
    mask = torch.rand_like(t) < ratio
    t[mask] = float("nan")
    return t


class NansumBenchmark(Benchmark):
    """
    Class for benchmarking sum operation.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        more_shapes_1d = [
            (1025 * 1024,),
            (1024 * 1024 * 1024,),
        ]
        more_shapes_2d = [(1024, 2**i) for i in range(0, 21, 4)]
        more_shapes_3d = [(64, 2**i, 64) for i in range(0, 15, 4)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d
        

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            inp = _inject_nans_(inp)
            if inp.ndim > 1:
                yield inp, 1
            else:
                yield inp,


@pytest.mark.nansum
def test_perf_nansum():
    bench = NansumBenchmark(op_name="nansum", torch_op=torch.nansum, dtypes=FLOAT_DTYPES)
    bench.run()

@pytest.mark.nansum
def test_perf_nansum_backward():
    bench = NansumBenchmark(op_name="nansum", torch_op=torch.nansum, dtypes=FLOAT_DTYPES, is_backward=True)
    bench.run()
