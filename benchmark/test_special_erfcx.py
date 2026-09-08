import pytest
import torch

from . import base


@pytest.mark.special_erfcx
def test_special_erfcx():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_erfcx",
        torch_op=torch.special.erfcx,
        # erfcx only supports float32
        dtypes=[torch.float32],
    )
    bench.run()
