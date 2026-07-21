import pytest
import torch

from . import base


@pytest.mark.special_modified_bessel_i1
def test_special_modified_bessel_i1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_modified_bessel_i1",
        torch_op=torch.special.modified_bessel_i1,
        # special.modified_bessel_i1 only supports float32/float64, not fp16/bf16
        dtypes=[torch.float32],
    )
    bench.run()
