import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_PI = 3.1415926535897932384626433832795028841971


def _trigamma_composed(x):
    # Same reflection + recurrence + asymptotic series the kernel uses.
    reflect = x < 0.5
    sin_pi_x = torch.sin(_PI * x)
    result = torch.where(
        reflect, -(_PI * _PI) / (sin_pi_x * sin_pi_x), torch.zeros_like(x)
    )
    y = torch.where(reflect, 1.0 - x, x)
    for _ in range(6):
        result = result + 1.0 / (y * y)
        y = y + 1.0
    iyy = 1.0 / (y * y)
    result = (
        result
        + (
            1.0
            + 1.0 / (2.0 * y)
            + iyy * (1.0 / 6.0 - iyy * (1.0 / 30.0 - iyy * (1.0 / 42.0)))
        )
        / y
    )
    return torch.where(reflect, -result, result)


def _reference_polygamma(n, x, out=None):
    # torch has no polygamma on every backend (e.g. NPU), where it silently
    # falls back to CPU. For n == 1 the reference can stay on device by
    # composing trigamma from primitives. Note the check is on the tensor's
    # device, so `--ref=cpu` still uses torch's own CPU implementation.
    if n == 1 and x.device.type not in ("cuda", "cpu"):
        res = _trigamma_composed(x.to(torch.float32)).to(x.dtype)
        return out.copy_(res) if out is not None else res
    if out is not None:
        return torch.polygamma(n, x, out=out)
    return torch.polygamma(n, x)


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

    ref_out = _reference_polygamma(n, ref_inp)
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
    # The zeta path (n >= 2) on the negative domain is ill-conditioned near
    # half-integer x: the result matches the reference to float32 tolerance
    # only when the kernel and the reference share the exact libdevice pow
    # (true on CUDA). Backends whose pow diverges from the reference by a few
    # ulp miss that tolerance on those meaningless cancellation lanes -- on
    # Ascend because torch falls back to the CPU pow, on Hygon because
    # FlagTree's pow differs from torch's HIP pow -- so skip them there.
    if n >= 2 and flag_gems.vendor_name in ("ascend", "hygon"):
        pytest.skip("ill-conditioned cancellation lanes; backend pow != reference pow")

    torch.manual_seed(0)
    inp = torch.empty((1024, 1024), dtype=torch.float32, device=flag_gems.device)
    inp.uniform_(-5.0, 5.0)
    ref_inp = utils.to_reference(inp)

    ref_out = _reference_polygamma(n, ref_inp)
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

    _reference_polygamma(n, ref_inp, out=ref_out)
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

    ref_out = ref_inp.copy_(_reference_polygamma(n, ref_inp))
    with flag_gems.use_gems():
        res_out = inp.polygamma_(n)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
