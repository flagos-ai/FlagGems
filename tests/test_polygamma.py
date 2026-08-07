import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# n = 0 and n = 1 hit the dedicated digamma / trigamma kernels; n >= 2 hits
# the Hurwitz zeta kernel. n = 8 keeps n! * zeta(n + 1, x) within float16
# range on the [1, 2) input domain.
POLYGAMMA_N = [0, 1, 2, 5, 8]


@pytest.mark.polygamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("n", POLYGAMMA_N)
def test_polygamma(shape, dtype, n):
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.polygamma(n, ref_inp)
    with flag_gems.use_gems():
        res_out = torch.polygamma(n, inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Even n >= 8 is excluded on the negative domain: there zeta's direct sum
# cancels catastrophically at half-integer x (odd s = n + 1 makes the
# (+-0.5)^-s pair terms cancel), which amplifies benign 1-ulp powf
# differences between Triton and torch far beyond float32 tolerance —
# torch itself is equally inaccurate vs exact math on those points.


@pytest.mark.polygamma
@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_polygamma_wide_domain(n):
    torch.manual_seed(0)
    inp = torch.empty((1024, 1024), dtype=torch.float32, device=flag_gems.device)
    inp.uniform_(-5.0, 5.0)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.polygamma(n, ref_inp)
    with flag_gems.use_gems():
        res_out = torch.polygamma(n, inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.polygamma_out
@pytest.mark.parametrize("n", [0, 1, 2])
def test_polygamma_out(n):
    torch.manual_seed(0)
    inp = torch.rand((1024, 1024), dtype=torch.float32, device=flag_gems.device) + 1.0
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.polygamma(n, ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.polygamma(n, inp, out=out)

    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.polygamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("n", POLYGAMMA_N)
def test_polygamma_(shape, dtype, n):
    torch.manual_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.polygamma_(n)
    with flag_gems.use_gems():
        res_out = inp.polygamma_(n)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
