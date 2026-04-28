import pytest
import torch

from . import base, consts


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES is included (Issue #2693).
@pytest.mark.mul
def test_mul():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
        # dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.mul_
def test_mul_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul_",
        torch_op=lambda a, b: a.mul_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
