import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.imag
def test_imag():
    """Benchmark imag operator performance on complex tensors."""
    bench = base.UnaryPointwiseBenchmark(
        op_name="imag",
        torch_op=torch.imag,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.set_gems(flag_gems.imag)
    bench.run()
