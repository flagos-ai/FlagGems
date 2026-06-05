import pytest
import torch

from . import base, consts, utils


def _input_fn_dim0(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": 0,
    },


def _input_fn_dim1(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "sorted": True,
        "return_inverse": True,
        "return_counts": False,
        "dim": 1,
    },


@pytest.mark.unique_dim
def test_unique_dim_dim0():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn_dim0,
        op_name="unique_dim",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.unique_dim
def test_unique_dim_dim1():
    bench = base.GenericBenchmark2DOnly(
        input_fn=_input_fn_dim1,
        op_name="unique_dim",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()
