import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.amin
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="amin",
        torch_op=torch.amin,
        dtypes=[dtype],
    )
    bench.run()


def amin_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = 1 if len(shape) > 1 else 0
    yield inp, dim


@pytest.mark.amin
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin_dim(dtype):
    bench = base.GenericBenchmark(
        op_name="amin_dim",
        torch_op=torch.amin,
        input_fn=amin_dim_input_fn,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.amin_
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin_(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="amin_",
        torch_op=lambda *a: a[0].copy_(torch.amin(*a, keepdim=True)),
        dtypes=[dtype],
        is_inplace=True,
        gems_op=flag_gems.amin_,
    )
    bench.run()
