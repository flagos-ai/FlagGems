import pytest
import torch

from . import base


@pytest.mark.special_gammainc
def test_special_gammainc():
    bench = base.BinaryPointwiseBenchmark(
        op_name="special_gammainc",
        torch_op=torch.special.gammainc,
        # float32 only: gammainc series expansion is numerically unstable in lower precisions
        dtypes=[torch.float32],
    )
    bench.run()
