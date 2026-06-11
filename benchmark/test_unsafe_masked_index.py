import pytest
import torch

from . import base, consts

# 1D shapes ranging from 1K to 1M elements for masked index coverage
UNSAFE_MASKED_INDEX_SHAPES = [
    (1024,),
    (2048,),
    (4096,),
    (8192,),
    (16384,),
    (32768,),
    (65536,),
    (131072,),
    (262144,),
    (1048576,),
]


class UnsafeMaskedIndexBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = UNSAFE_MASKED_INDEX_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            mask = torch.rand(shape, device=self.device) > 0.3
            indices = torch.randint(0, max(n, 1), shape, device=self.device)
            fill = 0.0
            yield inp, mask, [indices], fill


@pytest.mark.unsafe_masked_index
def test_unsafe_masked_index():
    bench = UnsafeMaskedIndexBenchmark(
        op_name="unsafe_masked_index",
        torch_op=torch._unsafe_masked_index,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
