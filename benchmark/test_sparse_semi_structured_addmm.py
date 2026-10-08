import pytest
import torch

import flag_gems

from . import base, consts

# Sparse semi-structured addmm shapes
SPARSE_SEMI_STRUCTURED_ADDMM_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 128),
    (512, 512),
]


def _sparse_semi_structured_addmm_ref(
    input_tensor: torch.Tensor,
    mat1: torch.Tensor,
    mat1_meta: torch.Tensor,
    mat2: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """Pure-PyTorch reference implementation.

    The native aten op uses a different input format and a different
    algorithm than the Gems implementation, so the native op cannot serve as
    an equivalent baseline. This reference shares both the input format and
    the algorithm with the Gems op, so it is used as the benchmark baseline.
    """
    M, K4 = mat1_meta.shape
    N = mat2.shape[1]

    # Reshape mat1 to (M, K4, 4)
    mat1_reshaped = mat1.view(M, K4, 4)

    # Reshape mat2 to (K4, 4, N)
    mat2_reshaped = mat2.view(K4, 4, N)

    # Create meta mask: for each (m, k), if meta[m,k] is True, keep pos 0,1,
    # else keep pos 2,3
    meta_mask = torch.cat(
        [
            mat1_meta.unsqueeze(2),
            mat1_meta.unsqueeze(2),
            (~mat1_meta).unsqueeze(2),
            (~mat1_meta).unsqueeze(2),
        ],
        dim=2,
    )

    # Apply mask to mat1_reshaped
    mat1_masked = torch.where(meta_mask, mat1_reshaped, torch.zeros_like(mat1_reshaped))

    # Compute result with masked values
    result = torch.zeros((M, N), dtype=mat1.dtype, device=mat1.device)
    for pos in range(4):
        a = mat1_masked[:, :, pos]  # (M, K4)
        b = mat2_reshaped[:, pos, :]  # (K4, N)
        result += torch.mm(a, b)

    # Apply alpha and beta scaling
    result = alpha * result + beta * input_tensor

    return result


class SparseSemiStructuredAddmmBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_SEMI_STRUCTURED_ADDMM_SHAPES

    def get_input_iter(self, cur_dtype):
        K4 = 32  # K = 4 * K4
        for shape in self.shapes:
            M, N = shape
            input_tensor = torch.randn(M, N, dtype=cur_dtype, device=self.device)
            mat1 = torch.randn(M, 4 * K4, dtype=cur_dtype, device=self.device)
            mat1_meta = torch.randint(
                0, 2, (M, K4), dtype=torch.bool, device=self.device
            )
            mat2 = torch.randn(4 * K4, N, dtype=cur_dtype, device=self.device)
            yield input_tensor, mat1, mat1_meta, mat2


@pytest.mark.sparse_semi_structured_addmm
def test_sparse_semi_structured_addmm():
    bench = SparseSemiStructuredAddmmBenchmark(
        op_name="sparse_semi_structured_addmm",
        torch_op=_sparse_semi_structured_addmm_ref,
        gems_op=flag_gems._sparse_semi_structured_addmm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
