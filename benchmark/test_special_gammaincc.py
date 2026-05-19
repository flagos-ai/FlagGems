import pytest
import torch

from . import base


@pytest.mark.special_gammaincc
def test_special_gammaincc():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_gammaincc",
        torch_op=torch.special.gammaincc,
        # igammac_cuda is not implemented for Half/BFloat16
        dtypes=[torch.float32],
    )
    bench.run()
