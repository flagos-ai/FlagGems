import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input


class DiffBenchmark(Benchmark):
    """Benchmark for torch.diff operation."""

    def set_more_shapes(self):
        more_shapes_1d = [
            (1024,),
            (1024 * 1024,),
            (16 * 1024 * 1024,),
        ]
        more_shapes_2d = [
            (1024, 1024),
            (4096, 4096),
            (8192, 8192),
        ]
        more_shapes_3d = [
            (64, 256, 256),
            (128, 512, 512),
        ]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield (inp,)


@pytest.mark.diff
def test_perf_diff():
    bench = DiffBenchmark(
        op_name="diff",
        torch_op=torch.diff,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.diff)
    bench.run()
