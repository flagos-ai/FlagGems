import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.eye
@pytest.mark.parametrize(
    "shape",
    [
        (256, 1024),
        (1024, 256),
        (8192, 4096),
        (4096, 8192),
    ]
    + [(2**d, 2**d) for d in range(7, 13)],
)
@pytest.mark.parametrize(
    "dtype", utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES + utils.BOOL_TYPES
)
def test_eye(shape, dtype):
    n, m = shape

    # test eye(n, m) without dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.eye(n, m, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out,
        torch.eye(n, m, dtype=dtype, device="cpu" if cfg.TO_CPU else device),
    )

    # test eye(n)
    with flag_gems.use_gems():
        res_out = torch.eye(n, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.eye(n, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out,
        torch.eye(n, dtype=dtype, device="cpu" if cfg.TO_CPU else device),
    )


@pytest.mark.eye_m
@pytest.mark.parametrize(
    "shape",
    [
        (256, 1024),
        (1024, 256),
        (8192, 4096),
        (4096, 8192),
    ]
    + [(2**d, 2**d) for d in range(7, 13)],
)
@pytest.mark.parametrize(
    "dtype", utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES + utils.BOOL_TYPES
)
def test_eye_m(shape, dtype):
    n, m = shape

    # test eye(n, m) without dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.eye(n, m, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.eye(n, m, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out,
        torch.eye(n, m, dtype=dtype, device="cpu" if cfg.TO_CPU else device),
    )
