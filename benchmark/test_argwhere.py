import pytest
import torch

from . import base, utils

# argwhere 主要受益于大型半精度 tensor；float32 由 torch 的 CUB 路径主导，
# 因此只对实际能获得加速的 dtype 进行 benchmark。
ARGWHERE_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.argwhere
def test_argwhere():
    bench = base.GenericBenchmark2DOnly(
        input_fn=utils.unary_input_fn,
        op_name="argwhere",
        torch_op=torch.argwhere,
        dtypes=ARGWHERE_DTYPES,
    )
    bench.run()
