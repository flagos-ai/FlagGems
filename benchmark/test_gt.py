import pytest
import torch

from . import base, consts, utils


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.gt
def test_gt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.gt_scalar
def test_gt_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="gt.Scalar",
        torch_op=torch.gt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
