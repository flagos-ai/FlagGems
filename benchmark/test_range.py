import pytest
import torch

from . import base

# Range is a non-pointwise op -- we override set_shapes instead of using
# GenericBenchmark which assumes pointwise tensor inputs.
RANGE_SIZES = [4096, 16777216, 1073741824]


class RangeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = RANGE_SIZES

    def get_input_iter(self, cur_dtype):
        for end in self.shapes:
            yield {"start": 0, "end": end, "dtype": cur_dtype},


@pytest.mark.range
def test_range():
    # torch.range does not support bfloat16 on CUDA
    dtypes = [torch.float16, torch.float32]
    bench = RangeBenchmark(
        op_name="range",
        torch_op=torch.range,
        dtypes=dtypes,
    )
    bench.run()
