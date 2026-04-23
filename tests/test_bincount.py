import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    ATTN_HEADS = [2]
    FLOAT_DTYPES = [torch.float32]
else:
    ATTN_HEADS = [2, 4, 8, 16, 32]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

BINCOUNT_SHAPES = [(16,), (4096,), (100000,)]
NUM_CLASSES_LIST = [10, 256]

# Make sure every thread has same seed.
random.seed(time.time() // 100)


def _assert_bincount(res_out, ref_out, dtype=None, shape=None, num_classes=None):
    if dtype is None:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        atol = (
            1e-3
            if (dtype == torch.float32 and shape[0] >= 100000 and num_classes <= 10)
            else 1e-4
        )
        utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.bincount
@pytest.mark.parametrize("shape", BINCOUNT_SHAPES)
@pytest.mark.parametrize("num_classes", NUM_CLASSES_LIST)
def test_bincount(shape, num_classes):
    inp = torch.randint(
        0, num_classes, shape, dtype=torch.int64, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    res_out = flag_gems.bincount(inp)

    _assert_bincount(res_out, ref_out)


@pytest.mark.bincount
@pytest.mark.parametrize("shape", BINCOUNT_SHAPES)
@pytest.mark.parametrize("num_classes", NUM_CLASSES_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bincount_weighted(shape, num_classes, dtype):
    inp = torch.randint(
        0, num_classes, shape, dtype=torch.int64, device=flag_gems.device
    )
    weights = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp, ref_weights = utils.to_reference(inp), utils.to_reference(weights)

    ref_out = torch.bincount(ref_inp, weights=ref_weights)
    res_out = flag_gems.bincount(inp, weights=weights)

    _assert_bincount(res_out, ref_out, dtype, shape, num_classes)


@pytest.mark.bincount
@pytest.mark.parametrize("shape", BINCOUNT_SHAPES)
@pytest.mark.parametrize("num_classes", NUM_CLASSES_LIST)
@pytest.mark.parametrize("minlength", [0, 512])
def test_bincount_minlength(shape, num_classes, minlength):
    inp = torch.randint(
        0, num_classes, shape, dtype=torch.int64, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp, minlength=minlength)
    res_out = flag_gems.bincount(inp, minlength=minlength)
    _assert_bincount(res_out, ref_out)

    dtype = torch.float32
    weights = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_weights = utils.to_reference(weights)

    ref_out_w = torch.bincount(ref_inp, weights=ref_weights, minlength=minlength)
    res_out_w = flag_gems.bincount(inp, weights=weights, minlength=minlength)

    _assert_bincount(res_out_w, ref_out_w, dtype, shape, num_classes)
