import pytest
import torch

from . import base, consts

# Custom shapes for _chunk_cat benchmark: 1D and 2D tensors
_CHUNK_CAT_SHAPES = [
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
]


class ChunkCatBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _CHUNK_CAT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            for num_chunks in [2, 4]:
                # Create input tensor
                inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
                yield [inp], 0, num_chunks


@pytest.mark.chunk_cat
def test_chunk_cat():
    bench = ChunkCatBenchmark(
        op_name="chunk_cat",
        torch_op=torch._chunk_cat,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
