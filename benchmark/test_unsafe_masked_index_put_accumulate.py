import pytest
import torch

from . import base

_UNSAFE_MASKED_INDEX_PUT_ACCUMULATE_SHAPES = [
    (10,),
    (20,),
    (50,),
    (100,),
]


class UnsafeMaskedIndexPutAccumulateBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _UNSAFE_MASKED_INDEX_PUT_ACCUMULATE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            mask = torch.rand(shape) < 0.3
            mask = mask.to(self.device)
            indices = torch.randint(
                0, max(shape[-1], 1), shape, dtype=torch.long, device=self.device
            )
            values = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, mask, (indices,), values


@pytest.mark.unsafe_masked_index_put_accumulate
def test_perf_unsafe_masked_index_put_accumulate():
    bench = UnsafeMaskedIndexPutAccumulateBenchmark(
        op_name="unsafe_masked_index_put_accumulate",
        torch_op=torch._unsafe_masked_index_put_accumulate,
        # tl.atomic_add does not support bfloat16; only float16/float32 are valid
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
