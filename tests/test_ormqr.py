import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import _SKIP_FP64

# Shapes for ormqr: (M, N) pairs covering square and rectangular matrices
ORMQR_SHAPES = [
    (32, 32),
    (64, 64),
    (128, 64),
    (64, 128),
]

# ormqr only supports float32 and float64 (LAPACK limitation, no half/bfloat16)
ORMQR_DTYPES = [torch.float32, pytest.param(torch.float64, marks=_SKIP_FP64)]


def _ormqr_cpu_reference(inp, tau, other, *, left, transpose):
    # Apply the packed Householder reflectors without calling CPU LAPACK.
    # Q = H0 H1 ...; left multiplication reverses the application order.
    indices = range(tau.shape[-1])
    if left != transpose:
        indices = reversed(indices)
    result = other.clone()
    for i in indices:
        tail = inp[..., i + 1 :, i]
        v = torch.cat((torch.ones_like(inp[..., i : i + 1, i]), tail), dim=-1)
        scale = tau[..., i, None, None]
        if left:
            block = result[..., i:, :]
            projection = v.unsqueeze(-2) @ block
            block.sub_(scale * (v.unsqueeze(-1) @ projection))
        else:
            block = result[..., :, i:]
            projection = block @ v.unsqueeze(-1)
            block.sub_(scale * (projection @ v.unsqueeze(-2)))
    return result


@pytest.mark.ormqr
@pytest.mark.parametrize("M, N", ORMQR_SHAPES)
@pytest.mark.parametrize("dtype", ORMQR_DTYPES)
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_ormqr(M, N, dtype, left, transpose):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 ormqr test on tsingmicro platform")
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Not supported float64")

    # For ormqr, we need valid Householder reflectors
    # Generate them from a QR decomposition (geqrf)
    k = min(M, N)

    # The matrix dimension depends on 'left' parameter
    # If left=True, we need (M, k); if left=False, we need (N, k)
    if left:
        a = torch.randn((M, k), dtype=dtype, device=flag_gems.device)
    else:
        a = torch.randn((N, k), dtype=dtype, device=flag_gems.device)

    # Generate Householder reflectors via QR decomposition
    input_tensor, tau = torch.geqrf(a)

    # The other matrix to multiply with - always has shape (M, N)
    other = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_tau = utils.to_reference(tau, True)
    ref_other = utils.to_reference(other, True)

    reference = torch.ormqr
    if (
        flag_gems.vendor_name == "mthreads"
        and ref_input.device.type == "cpu"
        and not torch._C.has_lapack
    ):
        reference = _ormqr_cpu_reference
    ref_out = reference(ref_input, ref_tau, ref_other, left=left, transpose=transpose)
    res_out = flag_gems.ormqr(input_tensor, tau, other, left=left, transpose=transpose)

    utils.gems_assert_close(res_out, ref_out, dtype)
