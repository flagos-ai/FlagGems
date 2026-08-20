import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _is_sm80_to_sm89():
    """Check whether the current device can run the aten op.

    The C++ kernel of ``aten._sparse_semi_structured_addmm`` only supports GPUs
    with compute capability 8.x. ``major == 8`` is the exact check performed by
    the kernel, so sm_70 / sm_90 / CPU / non-NVIDIA backends all return False.
    """
    if not torch.cuda.is_available() or flag_gems.device != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 8


def _pr_meta_to_packed(mat1, meta_bool):
    """Convert the PR (M, K4) boolean meta together with the dense ``mat1``
    into the ``SparseSemiStructuredTensor`` expected by the aten op.

    PR semantics: meta=True keeps positions [0, 1] of each 4-element K group,
    meta=False keeps positions [2, 3]. The dense mat1 is masked accordingly
    and packed via ``torch.sparse.to_sparse_semi_structured``.
    """
    M, K4 = meta_bool.shape
    group_pos = torch.arange(4 * K4, device=mat1.device) % 4
    k_of = torch.arange(4 * K4, device=mat1.device) // 4
    meta_expanded = meta_bool[:, k_of]
    keep_mask = (group_pos[None, :] < 2) == meta_expanded
    mat1_sparse = torch.where(keep_mask, mat1, torch.zeros_like(mat1))
    return torch.sparse.to_sparse_semi_structured(mat1_sparse)


def _aten_addmm_ref(input_tensor, mat1, meta, mat2, alpha=1.0, beta=1.0):
    """Reference output from the native ``aten._sparse_semi_structured_addmm``.

    Since PyTorch 2.11 the aten op follows a sparse-linear style schema:
    ``input`` (the first positional argument) must be a 1D bias vector of
    length ``M`` (broadcast across columns), while this operator computes a
    2D addmm ``alpha * (mat1 @ sparse(meta)) @ mat2 + beta * input`` with an
    ``(M, N)`` input applied element-wise. The two are not equivalent for a
    general 2D input, so the reference is split into two parts:

    1. An empty input tensor is passed to the aten op to select the
       no-bias epilogue path, which computes ``alpha * packed @ mat2`` only.
       (The input dimension/size checks in the C++ kernel are gated on
       ``input.numel() != 0``, so an empty tensor bypasses them.)
    2. The ``beta * input`` contribution is added back in Python so the
       full addmm semantics are reproduced.

    The CUTLASS backend is used so that ``packed.meta`` is a standalone int16
    tensor matching the ``Tensor mat1_meta`` parameter of the aten op schema.

    Must run on GPU since the CUTLASS backend does not support CPU.
    """
    old = torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS
    torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
    try:
        packed = _pr_meta_to_packed(mat1, meta)
        # Empty input selects the no-bias epilogue: the aten op then only
        # computes alpha * packed @ mat2 (with the 2:4 sparsity applied via meta).
        empty_input = torch.zeros(0, dtype=mat1.dtype, device=mat1.device)
        mm_out = torch.ops.aten._sparse_semi_structured_addmm(
            empty_input,
            packed.packed,
            packed.meta,
            mat2,
            alpha=alpha,
            beta=1.0,
        )
        # Re-apply the 2D input element-wise to restore addmm semantics.
        return mm_out + beta * input_tensor
    finally:
        torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = old


@pytest.mark.sparse_semi_structured_addmm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128), (256, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_addmm(shape, dtype):
    if not _is_sm80_to_sm89():
        pytest.skip("aten._sparse_semi_structured_addmm only supports 8.x")

    # float32 has no native backend that can represent 2:4 sparsity, so the
    # aten reference is unavailable and the (correct) gems kernel is skipped.
    if dtype == torch.float32:
        pytest.skip("float32 2:4 sparsity has no supported native backend")

    M, N = shape
    K4 = 32  # K = 4 * K4

    input_tensor = torch.randn(M, N, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    ref_out = _aten_addmm_ref(input_tensor, mat1, mat1_meta, mat2)

    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_addmm(
            input_tensor, mat1, mat1_meta, mat2
        )

    # NOTE: re-measure on sm_80/86/89 hardware since the CUTLASS C++ kernel is
    # not reachable on sm_90 (compute capability 8.x required).
    # In quick-cpu mode (TO_CPU) the CUTLASS ref must stay on GPU, so move only
    # the final ref_out to CPU here to honour the to_cpu() contract.
    ref_out = utils.to_reference(ref_out)
    if dtype == torch.bfloat16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.15)
    elif dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.08)
    else:  # float32
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.02)


@pytest.mark.sparse_semi_structured_addmm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_addmm_with_alpha_beta(shape, dtype):
    if not _is_sm80_to_sm89():
        pytest.skip("aten._sparse_semi_structured_addmm only supports 8.x")

    # float32 has no native backend that can represent 2:4 sparsity, so the
    # aten reference is unavailable and the (correct) gems kernel is skipped.
    if dtype == torch.float32:
        pytest.skip("float32 2:4 sparsity has no supported native backend")

    M, N = shape
    K4 = 32  # K = 4 * K4
    alpha = 2.5
    beta = 0.5

    input_tensor = torch.randn(M, N, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    ref_out = _aten_addmm_ref(
        input_tensor, mat1, mat1_meta, mat2, alpha=alpha, beta=beta
    )

    with flag_gems.use_gems():
        res_out = flag_gems._sparse_semi_structured_addmm(
            input_tensor, mat1, mat1_meta, mat2, alpha=alpha, beta=beta
        )

    # NOTE: re-measure on sm_80/86/89 hardware since the CUTLASS C++ kernel is
    # not reachable on sm_90 (compute capability 8.x required).
    # In quick-cpu mode (TO_CPU) the CUTLASS ref must stay on GPU, so move only
    # the final ref_out to CPU here to honour the to_cpu() contract.
    ref_out = utils.to_reference(ref_out)
    if dtype == torch.bfloat16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.4)
    elif dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.2)
    else:  # float32
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.02)
