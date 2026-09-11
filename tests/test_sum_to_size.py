import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Each case is (input_shape, target_size). ``target_size`` must be broadcastable
# to ``input_shape``; sum_to_size reverses the broadcast by summing.
SUM_TO_SIZE_CASES = [
    ((1024, 1024), (1, 1024)),
    ((1024, 1024), (1024, 1)),
    ((1024, 1024), (1, 1)),
    ((1024, 1024), (1024, 1024)),
    ((20, 320, 15), (320, 15)),
    ((20, 320, 15), (1, 320, 15)),
    ((20, 320, 15), (20, 1, 15)),
    ((20, 320, 15), (1, 1, 1)),
    ((16, 128, 64, 60), (64, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 128, 1, 1)),
    ((8192,), (1,)),
    ((32, 50257), (1, 50257)),
    ((32, 50257), (32, 1)),
]


@pytest.mark.sum_to_size
@pytest.mark.parametrize("shape, size", SUM_TO_SIZE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sum_to_size(shape, size, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = ref_inp.sum_to_size(size)
    res_out = flag_gems.sum_to_size(res_inp, size)

    # Reduction length drives accumulation error, so scale tolerance by it.
    reduce_len = res_inp.numel() // int(torch.tensor(size).prod().item())
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_len)


@pytest.mark.sum_to_size
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sum_to_size_extreme(dtype):
    # A moderate 2D shape reduced along dim 0: enough rows to exercise the
    # accumulation path while keeping the case cheap for edge-value checks.
    shape = (64, 1024)
    size = (1, 1024)

    # All-zero input.
    res_inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_out = ref_inp.sum_to_size(size)
    res_out = flag_gems.sum_to_size(res_inp, size)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[0])

    # Uniform-value input.
    res_inp = torch.full(shape, 2.0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_out = ref_inp.sum_to_size(size)
    res_out = flag_gems.sum_to_size(res_inp, size)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[0])
