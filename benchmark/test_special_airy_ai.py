import pytest
import torch

from . import base


@pytest.mark.special_airy_ai
def test_special_airy_ai():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_airy_ai",
        torch_op=torch.special.airy_ai,
        # torch.special.airy_ai only supports float32 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_airy_ai_out
def test_special_airy_ai_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_airy_ai_out",
        torch_op=torch.special.airy_ai,
        # torch.special.airy_ai only supports float32 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
