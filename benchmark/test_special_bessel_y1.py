import pytest
import torch

from . import base


@pytest.mark.special_bessel_y1
def test_special_bessel_y1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_y1",
        torch_op=torch.special.bessel_y1,
        # bessel_y1_cuda only supports float32; Half/BFloat16 raise RuntimeError
        dtypes=[torch.float32],
    )
    bench.run()
