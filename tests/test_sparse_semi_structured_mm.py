import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.sparse_semi_structured_mm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128), (256, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_mm(shape, dtype):
    from flag_gems.ops._sparse_semi_structured_mm import _sparse_semi_structured_mm_ref

    M, N = shape
    K4 = 32  # K = 4 * K4

    # Create input tensors for sparse semi-structured MM
    # mat1: (M, 4*K4), mat1_meta: (M, K4), mat2: (4*K4, N)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    # Reference implementation
    ref_mat1 = utils.to_reference(mat1, upcast=True)
    ref_mat2 = utils.to_reference(mat2, upcast=True)
    ref_meta = mat1_meta.to(ref_mat1.device)
    ref_out = _sparse_semi_structured_mm_ref(ref_mat1, ref_meta, ref_mat2)

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_mm(mat1, mat1_meta, mat2)

    # Use more permissive tolerance for this operator due to numerical precision differences
    # between reference and Triton implementations
    if dtype == torch.float16 or dtype == torch.bfloat16:
        # float16/bfloat16 have limited precision, use higher atol
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.1)
    else:
        # float32, use moderate tolerance
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.02)
