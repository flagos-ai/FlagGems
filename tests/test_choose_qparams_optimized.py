import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# choose_qparams_optimized is float32-only in the aten reference and only has a
# CPU implementation, so we always compare against the CPU reference.
# The op makes discrete branch decisions by comparing float32 sums of squared
# quantization errors. CPU (sequential fp32) and GPU (tree reduction) summation
# diverge as the element count grows, which can flip the search and yield a
# different-but-equally-valid range. We therefore test element counts where the
# decisions are numerically stable across both reduction orders.
CQO_SHAPES = [(128,), (256,), (512,), (1024,), (64, 16)]
# choose_qparams_optimized is float32-only in the aten reference (CPU-only impl),
# so the tested dtype is restricted to float32 rather than utils.FLOAT_DTYPES.
CQO_DTYPES = [torch.float32]


@pytest.mark.choose_qparams_optimized
@pytest.mark.parametrize("shape", CQO_SHAPES)
@pytest.mark.parametrize("dtype", CQO_DTYPES)
@pytest.mark.parametrize("n_bins", [200, 100])
@pytest.mark.parametrize("ratio", [0.16, 0.5])
def test_choose_qparams_optimized(shape, dtype, n_bins, ratio):
    bit_width = 8
    torch.manual_seed(0)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    numel = res_inp.numel()

    # aten reference is CPU-only float32; force CPU placement regardless of --ref.
    ref_inp = utils.to_reference(res_inp).cpu()
    ref_max, ref_min = torch.choose_qparams_optimized(
        ref_inp.view(-1), numel, n_bins, ratio, bit_width
    )
    # gems_assert_close moves res to CPU when TO_CPU and asserts ref is already on
    # CPU; only move the reference to device for the non-TO_CPU comparison path.
    if not utils.TO_CPU:
        ref_max = ref_max.to(flag_gems.device)
        ref_min = ref_min.to(flag_gems.device)

    with flag_gems.use_gems():
        res_max, res_min = torch.choose_qparams_optimized(
            res_inp.view(-1), numel, n_bins, ratio, bit_width
        )

    utils.gems_assert_close(res_max, ref_max, torch.float32)
    utils.gems_assert_close(res_min, ref_min, torch.float32)


@pytest.mark.choose_qparams_optimized
def test_choose_qparams_optimized_constant_input():
    # data_range == 0 branch: all values identical.
    bit_width = 8
    n_bins = 200
    ratio = 0.16
    res_inp = torch.full((512,), 1.5, dtype=torch.float32, device=flag_gems.device)
    numel = res_inp.numel()

    ref_inp = utils.to_reference(res_inp).cpu()
    ref_max, ref_min = torch.choose_qparams_optimized(
        ref_inp, numel, n_bins, ratio, bit_width
    )
    if not utils.TO_CPU:
        ref_max = ref_max.to(flag_gems.device)
        ref_min = ref_min.to(flag_gems.device)

    with flag_gems.use_gems():
        res_max, res_min = torch.choose_qparams_optimized(
            res_inp, numel, n_bins, ratio, bit_width
        )

    utils.gems_assert_close(res_max, ref_max, torch.float32)
    utils.gems_assert_close(res_min, ref_min, torch.float32)
