import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
    QUANTILE_Q = [(0.2, 0.5, 0.8)]
    QUANTILE_INTERPOLATION = ["linear"]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]
    QUANTILE_Q = [
        (0.4),
        (0.0, 0.2, 0.5, 0.8, 1.0),
        (0.662, 0.8, 0.104, 0.99, 0.347, 0.255),
    ]
    QUANTILE_INTERPOLATION = ["linear", "lower", "higher", "nearest", "midpoint"]

QUANTILE_SHAPES = utils.REDUCTION_SMALL_SHAPES + [(10, 64, 196), (65535, 1)]
QUANTILE_FLOAT_DTYPES = [torch.float32]


@pytest.mark.quantile
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"), reason="Skipping Triton version."
)
@pytest.mark.parametrize("shape", QUANTILE_SHAPES)
@pytest.mark.parametrize("dtype", QUANTILE_FLOAT_DTYPES)
@pytest.mark.parametrize("q", QUANTILE_Q)
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATION)
def test_quantile(shape, dtype, q, interpolation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    q = torch.tensor(q, dtype=dtype, device=inp.device)
    ref_q = utils.to_reference(q)

    ref_out = torch.quantile(ref_inp, ref_q, interpolation=interpolation)
    with flag_gems.use_gems():
        res_out = torch.quantile(inp, q, interpolation=interpolation)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.quantile
@pytest.mark.skipif(
    utils.SkipVersion("triton", "<3.0"), reason="Skipping Triton version."
)
@pytest.mark.parametrize("shape", QUANTILE_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", QUANTILE_FLOAT_DTYPES)
@pytest.mark.parametrize("q", QUANTILE_Q)
@pytest.mark.parametrize("interpolation", QUANTILE_INTERPOLATION)
def test_quantile_dim(shape, dim, keepdim, dtype, q, interpolation):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    q = torch.tensor(q, dtype=dtype, device=inp.device)
    ref_q = utils.to_reference(q)

    ref_out = torch.quantile(
        ref_inp, ref_q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    with flag_gems.use_gems():
        res_out = torch.quantile(
            inp, q, dim=dim, keepdim=keepdim, interpolation=interpolation
        )

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)
