import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for linalg_multi_dot: (dim0, dim1, dim2, ...) where consecutive dims must match
# e.g., (2, 3, 4) means: A @ B @ C where A: (2,3), B: (3,4), C: (4, ?)
MULTI_DOT_SHAPES = [
    (2, 3, 4),  # 3 matrices: (2,3) @ (3,4) -> result (2,4)
    (2, 3, 4, 5),  # 4 matrices: (2,3) @ (3,4) @ (4,5) -> result (2,5)
    (10, 20, 30),  # 3 matrices
    (10, 20, 30, 40),  # 4 matrices
    (16, 32, 64, 128),  # 4 matrices
]


@pytest.mark.linalg_multi_dot
@pytest.mark.parametrize("shape", MULTI_DOT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_multi_dot(shape, dtype):
    # Create list of matrices with matching dimensions
    tensors = []
    for i in range(len(shape) - 1):
        mat = torch.randn(shape[i], shape[i + 1], dtype=dtype, device=flag_gems.device)
        tensors.append(mat)

    ref_tensors = [utils.to_reference(t, True) for t in tensors]

    ref_out = torch.linalg.multi_dot(ref_tensors)
    with flag_gems.use_gems():
        res_out = torch.linalg.multi_dot(tensors)

    # Use higher tolerance for multi_dot since:
    # 1. Errors accumulate with multiple matrix multiplications
    # 2. Reference is computed in float64 which has higher precision than fp16/bf16
    # 3. GEMS mm uses Triton which may have different numerical behavior than cuBLAS
    # For larger shapes with more multiplications, use even higher tolerance
    num_multiplications = len(shape) - 2  # number of @ operations
    # Multiply by num_multiplications to account for error accumulation
    # Use even larger multiplier for bfloat16 since it has lower precision
    multiplier = 48 if dtype == torch.bfloat16 else 8
    reduce_dim = shape[1] * max(1, num_multiplications * multiplier)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
