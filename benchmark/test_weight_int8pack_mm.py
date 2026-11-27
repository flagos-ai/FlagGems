import pytest
import torch

from . import base

# FP16/BF16 only: int8 matmul requires half-precision activation
FP16_BF16_DTYPES = [torch.float16, torch.bfloat16]


# LLM-scale shapes: (M, N, K) where M = tokens, N = output features, K = input features.
# These cover typical weight matrix dimensions found in transformer models.
WEIGHT_INT8PACK_MM_SHAPES = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (1, 8192, 8192),
    (1, 8192, 28672),
    (1, 28672, 8192),
    (4, 4096, 4096),
    (4, 4096, 11008),
    (4, 11008, 4096),
    (16, 4096, 4096),
    (16, 4096, 11008),
    (16, 11008, 4096),
    (32, 4096, 4096),
    (32, 4096, 11008),
    (32, 11008, 4096),
    (64, 4096, 4096),
    (64, 4096, 11008),
    (64, 11008, 4096),
    (128, 4096, 4096),
    (128, 4096, 11008),
    (128, 11008, 4096),
]


class WeightInt8packMMBenchmark(base.Benchmark):


    def set_shapes(self, shape_file_path=None):
        self.shapes = WEIGHT_INT8PACK_MM_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            M, N, K = shape
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"A": [M, K], "B": [N, K], "scales": [N]},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        M, N, K = shape
        A = torch.randn((M, K), dtype=case.dtype, device=self.device)
        B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=self.device)
        scales = torch.randn((N,), dtype=case.dtype, device=self.device)
        return (A, B, scales)





@pytest.mark.weight_int8pack_mm
def test_weight_int8pack_mm():
    bench = WeightInt8packMMBenchmark(
        op_name="weight_int8pack_mm",
        torch_op=torch.ops.aten._weight_int8pack_mm.default,
        dtypes=FP16_BF16_DTYPES,
    )
    bench.run()
