import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The primary reference is the device-agnostic pure-PyTorch 2:4 implementation
# (``_pytorch_ref``): it folds the 2:4 select mask into ``mat1`` and runs a plain
# fp32-accumulated ``alpha * (mat1 @ mat2) + beta * input``, mirroring the gems
# kernel's own fp32 accumulation. Because it has no device or dtype constraints,
# the gems kernel is exercised on every platform FlagGems supports (NVIDIA
# SM8.x/SM9.x and newer, non-NVIDIA accelerators, and CPU under ``--ref=cpu``).
#
# The native ``aten._sparse_semi_structured_addmm`` (CUTLASS backend) is used
# only as an EXTRA hardware cross-check, and only where it actually runs: NVIDIA
# GPUs with compute capability 8.x (the C++ kernel hard-requires ``major == 8``;
# sm_70 / sm_90 / CPU / non-NVIDIA backends are unsupported). The native op never
# gates whether the test runs.


def _is_sm80_to_sm89():
    """Whether the current device can run the native aten op for a cross-check.

    The C++ kernel of ``aten._sparse_semi_structured_addmm`` only supports GPUs
    with compute capability 8.x. ``major == 8`` is the exact check performed by
    the kernel, so sm_70 / sm_90 / CPU / non-NVIDIA backends all return False.
    """
    if not torch.cuda.is_available() or flag_gems.device != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 8


def _apply_2_4_mask(mat1, meta_bool):
    """Zero out the non-kept entries of ``mat1`` per the PR 2:4 semantics.

    meta=True keeps positions [0, 1] of each 4-element K group; meta=False keeps
    positions [2, 3]. Returns a dense tensor with the dropped positions zeroed.
    """
    M, K = mat1.shape
    group_pos = torch.arange(K, device=mat1.device) % 4
    k_of = torch.arange(K, device=mat1.device) // 4
    meta_expanded = meta_bool[:, k_of]
    keep_mask = (group_pos[None, :] < 2) == meta_expanded
    return torch.where(keep_mask, mat1, torch.zeros_like(mat1))


def _pytorch_ref(input_tensor, mat1, meta, mat2, alpha=1.0, beta=1.0):
    """Pure-PyTorch 2:4 addmm reference (device-agnostic, primary reference).

    Computes ``alpha * (sparse(mat1) @ mat2) + beta * input`` and casts back to
    the input dtype. The matmul accumulates in high precision (fp64 where the
    device supports it, else fp32) so the reference is not degraded by TF32,
    which some GPUs enable for fp32 matmuls. The gems kernel does true-fp32
    scalar accumulation (TF32 never applies), so it must be validated against a
    clean reference rather than a TF32-contaminated one.
    """
    acc_dtype = torch.float64 if utils.fp64_is_supported else torch.float32
    mat1_sparse = _apply_2_4_mask(mat1, meta)
    out = alpha * (
        mat1_sparse.to(acc_dtype) @ mat2.to(acc_dtype)
    ) + beta * input_tensor.to(acc_dtype)
    return out.to(mat1.dtype)


def _pr_meta_to_packed(mat1, meta_bool):
    """Convert the PR (M, K4) boolean meta together with the dense ``mat1`` into
    the ``SparseSemiStructuredTensor`` expected by the aten op."""
    mat1_sparse = _apply_2_4_mask(mat1, meta_bool)
    return torch.sparse.to_sparse_semi_structured(mat1_sparse)


def _aten_addmm_ref(input_tensor, mat1, meta, mat2, alpha=1.0, beta=1.0):
    """Cross-check output from the native ``aten._sparse_semi_structured_addmm``.

    Since PyTorch 2.11 the aten op follows a sparse-linear style schema: the
    first positional ``input`` is a 1D bias vector, not the 2D (M, N) addend this
    operator uses. So the reference is split: an empty input selects the no-bias
    epilogue (``alpha * packed @ mat2`` only), and ``beta * input`` is added back
    in Python. The CUTLASS backend is forced so ``packed.meta`` is a standalone
    tensor matching the op's ``mat1_meta`` parameter. GPU only (no CPU backend).
    """
    old = torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS
    torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
    try:
        packed = _pr_meta_to_packed(mat1, meta)
        empty_input = torch.zeros(0, dtype=mat1.dtype, device=mat1.device)
        mm_out = torch.ops.aten._sparse_semi_structured_addmm(
            empty_input,
            packed.packed,
            packed.meta,
            mat2,
            alpha=alpha,
            beta=1.0,
        )
        return mm_out + beta * input_tensor
    finally:
        torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = old


def _maybe_cross_check_native(
    res_out, input_tensor, mat1, meta, mat2, *, dtype, alpha=1.0, beta=1.0
):
    """Extra hardware cross-check against the native aten op.

    Runs only on NVIDIA SM8.x and only for fp16/bf16 (the native 2:4 backends;
    float32 has no native 2:4 backend). Otherwise a no-op. Purely additive on top
    of the device-agnostic ``_pytorch_ref`` comparison the caller already did.
    """
    if not _is_sm80_to_sm89():
        return
    if dtype not in (torch.float16, torch.bfloat16):
        return
    nat = _aten_addmm_ref(input_tensor, mat1, meta, mat2, alpha=alpha, beta=beta)
    nat = utils.to_reference(nat)
    atol = 0.5 if dtype == torch.bfloat16 else 0.12
    utils.gems_assert_close(res_out, nat, dtype, atol=atol)


@pytest.mark.sparse_semi_structured_addmm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128), (256, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_addmm(shape, dtype):
    M, N = shape
    K4 = 32  # K = 4 * K4

    input_tensor = torch.randn(M, N, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    ref_out = _pytorch_ref(input_tensor, mat1, mat1_meta, mat2)
    res_out = flag_gems._sparse_semi_structured_addmm(
        input_tensor, mat1, mat1_meta, mat2
    )

    ref_out = utils.to_reference(ref_out)
    if dtype == torch.bfloat16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.2)
    elif dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.08)
    else:  # float32
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.02)

    _maybe_cross_check_native(res_out, input_tensor, mat1, mat1_meta, mat2, dtype=dtype)


@pytest.mark.sparse_semi_structured_addmm
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_semi_structured_addmm_with_alpha_beta(shape, dtype):
    M, N = shape
    K4 = 32  # K = 4 * K4
    alpha = 2.5
    beta = 0.5

    input_tensor = torch.randn(M, N, dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=flag_gems.device)
    mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=flag_gems.device)
    mat2 = torch.randn(4 * K4, N, dtype=dtype, device=flag_gems.device)

    ref_out = _pytorch_ref(input_tensor, mat1, mat1_meta, mat2, alpha=alpha, beta=beta)
    res_out = flag_gems._sparse_semi_structured_addmm(
        input_tensor, mat1, mat1_meta, mat2, alpha=alpha, beta=beta
    )

    ref_out = utils.to_reference(ref_out)
    if dtype == torch.bfloat16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.5)
    elif dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.12)
    else:  # float32
        utils.gems_assert_close(res_out, ref_out, dtype, atol=0.02)

    _maybe_cross_check_native(
        res_out,
        input_tensor,
        mat1,
        mat1_meta,
        mat2,
        dtype=dtype,
        alpha=alpha,
        beta=beta,
    )
