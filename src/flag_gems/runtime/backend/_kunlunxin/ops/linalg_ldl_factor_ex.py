import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

from .linalg_ldl_factor import MAX_MATRIX_SIZE, _ldl_factor_kernel


@libentry()
@triton.jit
def _ldl_init_info_kernel(info, num_batches):
    batch_idx = tl.program_id(0)
    if batch_idx < num_batches:
        tl.store(info + batch_idx, 0)


def ldl_factor_ex(A, hermitian=False, check_errors=False):
    """Kunlunxin LDL factorization for symmetric positive definite matrices."""
    n = A.shape[-1]
    matrix_size = n * n
    num_batches = A.numel() // matrix_size

    LD = torch.empty_like(A)
    pivots = torch.empty(*A.shape[:-1], dtype=torch.int32, device=A.device)
    info = torch.empty(A.shape[:-2], dtype=torch.int32, device=A.device)

    _ldl_factor_kernel[(num_batches,)](
        A,
        LD,
        pivots,
        n,
        MAX_SIZE=MAX_MATRIX_SIZE,
    )
    _ldl_init_info_kernel[(num_batches,)](info, num_batches)
    return LD, pivots, info
