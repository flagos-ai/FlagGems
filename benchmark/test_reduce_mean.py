from typing import Generator

import pytest

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts


class reduce_meanBenchmark(base.Benchmark):
    """
    Benchmark for reduce_mean operator.
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
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                yield inp, 1
            else:
                yield inp,

    def get_torch_op(self):
        # Use flag_gems.reduce_mean as the torch_op for benchmarking
        return flag_gems.reduce_mean

    def get_gems_op(self):
        return flag_gems.reduce_mean


@pytest.mark.reduce_mean
def test_reduce_mean():
    bench = reduce_meanBenchmark(
        op_name="reduce_mean",
        torch_op=flag_gems.reduce_mean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
