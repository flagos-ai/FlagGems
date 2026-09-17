import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def slogdet_post_kernel(
    LU_ptr,
    piv_ptr,
    sign_out,
    lad_out,
    n,
    stride_lu_b,
    stride_lu_r,
    stride_lu_c,
    stride_piv_b,
    stride_sign,
    stride_lad,
    BLOCK: tl.constexpr,
):
    # One program per batch matrix. Reads only the diagonal of the LU factor and
    # the pivot vector -> sign, logabsdet. No data-dependent addressing.
    pid = tle.program_id(0)
    LU_mat = LU_ptr + pid * stride_lu_b
    r = tl.arange(0, BLOCK)
    valid = r < n
    diag = tl.load(
        LU_mat + r * stride_lu_r + r * stride_lu_c, mask=valid, other=1.0
    ).to(tl.float32)
    da = tl.abs(diag)
    zero = valid & (da < 1e-12)
    is_singular = tl.max(tl.where(zero, 1, 0)) > 0
    da = tl.where(zero, 1.0, da)
    neg = tl.sum(tl.where(valid & (diag < 0), 1, 0))
    piv = tl.load(piv_ptr + pid * stride_piv_b + r, mask=valid, other=0).to(tl.int32)
    swap = tl.sum(tl.where(valid & (piv != (r + 1)), 1, 0))
    parity = (neg + swap) % 2
    sign = tl.where(parity == 1, -1.0, 1.0)
    lad = tl.sum(tl.where(valid, tl.log(da), 0.0))
    sign = tl.where(is_singular, 0.0, sign)
    lad = tl.where(is_singular, -float("inf"), lad)
    tl.store(sign_out + pid * stride_sign, sign.to(sign_out.dtype.element_ty))
    tl.store(lad_out + pid * stride_lad, lad.to(lad_out.dtype.element_ty))


def linalg_slogdet(A):
    """sign and log|det| of a square matrix.

    The generic Triton kernel does fully-serial scalar Gaussian elimination with a
    data-dependent row swap (``A + max_row * stride``); on the kunlunxin XPU that
    address pattern miscompiles (wrong results) and is also extremely slow. Here we
    delegate the factorization to the vendor's ``lu_factor`` (LAPACK-backed, not
    registered by gems so it stays on the backend) and fuse the sign/logabsdet
    reduction into a single Triton launch.
    """
    logger.debug("GEMS LINALG_SLOGDET")
    assert A.dtype == torch.float32, f"slogdet: unsupported dtype {A.dtype}"
    assert A.dim() >= 2 and A.shape[-1] == A.shape[-2], "Input must be square"

    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    batch_size = 1
    for d in batch_shape:
        batch_size *= d

    sign = torch.empty(batch_shape, dtype=A.dtype, device=A.device)
    logabsdet = torch.empty(batch_shape, dtype=A.dtype, device=A.device)

    if batch_size == 0:
        return torch.zeros_like(sign), torch.full_like(logabsdet, float("-inf"))

    LU, pivots, _info = torch.linalg.lu_factor_ex(A)
    LU = LU.reshape(batch_size, n, n)
    pivots = pivots.reshape(batch_size, n).to(torch.int32)
    sign_flat = sign.reshape(batch_size)
    lad_flat = logabsdet.reshape(batch_size)

    BLOCK = triton.next_power_of_2(n)
    with torch_device_fn.device(A.device):
        slogdet_post_kernel[(batch_size,)](
            LU,
            pivots,
            sign_flat,
            lad_flat,
            n,
            LU.stride(0),
            LU.stride(1),
            LU.stride(2),
            pivots.stride(0),
            sign_flat.stride(0),
            lad_flat.stride(0),
            BLOCK,
            num_warps=1,
        )

    return sign, logabsdet
