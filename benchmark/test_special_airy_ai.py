import pytest
import torch

from . import base


@pytest.mark.special_airy_ai
def test_special_airy_ai():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_airy_ai",
        torch_op=torch.special.airy_ai,
        # airy_ai_cuda does not support Half/BFloat16
        dtypes=[torch.float32],
    )
    bench.run()
