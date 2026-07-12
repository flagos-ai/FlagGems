import random

import numpy as np
import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


MK_SHAPES = (
    [(1, 32)]
    if QUICK_MODE
    else [
        (1, 32),
        (7, 33),
        (31, 65),
        (160, 1024),
        (257, 96),
        (1023, 255),
        (5333, 71),
    ]
)


# Issue #2833: fails at (1, 1, 2)
@pytest.mark.mm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_broadcast_stride_zero(dtype):
    """Regression test: broadcast tensors (stride=0) must not crash TMA path."""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working ")
    torch.manual_seed(0)
    M, K, N = 128, 256, 256

    # Simulate the stride=(0,0) tensor that autograd produces from sum().backward():
    # scalar expand -> all strides are 0
    a = torch.randn((), dtype=dtype, device=flag_gems.device).expand(M, K)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    assert a.stride() == (0, 0)

    ref_a = utils.to_reference(a.contiguous(), True)
    ref_b = utils.to_reference(b, True)

    ref_out = torch.mm(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


# issue #2489: unaligned-stride or unaligned-base operands hit the Hopper host-TMA
# descriptor, which requires 16-byte-aligned strides and base. Both sizes exercise
# that path; the larger one covers bigger block sizes. M, N and K are all multiples
# of 8 so that is_tma_compatible() holds and a descriptor is actually built.
_UNALIGNED_MNK = [(64, 64, 64), (256, 128, 256)]


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("M, N, K", _UNALIGNED_MNK)
@pytest.mark.parametrize("pad", [1, 8])
def test_mm_unaligned_stride(dtype, M, N, K, pad):
    """Regression test for #2489: mm on a row-major *view* whose outer (row)
    stride is not 16-byte aligned must be correct on Hopper.

    ``as_strided((M, K), (K + pad, 1))`` keeps the inner dim contiguous but gaps
    the rows. A 16B-*misaligned* row stride (pad=1, e.g. 65 * 2B = 130B) fed to the
    host-TMA ``TensorDescriptor`` raises "strides must be 16-byte aligned"; it is
    fixed by copying the operand to contiguous. A 16B-*aligned* gap (pad=8) stays on
    the fast TMA path and must remain numerically correct.
    """
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    stride0 = K + pad
    base = torch.randn(M * stride0 + K, dtype=dtype, device=flag_gems.device)
    a = torch.as_strided(base, (M, K), (stride0, 1))
    assert not a.is_contiguous() and a.stride() == (stride0, 1)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mm(
        utils.to_reference(a.contiguous(), True), utils.to_reference(b, True)
    )
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("pad", [1, 8])
def test_mm_unaligned_stride_column_major(dtype, pad):
    """#2489 for a column-major operand: the weight ``b`` is a gapped transpose
    view (stride ``(1, K + pad)``); an unaligned outer stride must also be handled
    (exercises the column-major branch of the fix)."""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 256, 128, 256
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    stride1 = K + pad
    base = torch.randn(N * stride1 + K, dtype=dtype, device=flag_gems.device)
    b = torch.as_strided(base, (K, N), (1, stride1))  # column-major gapped view
    assert b.stride() == (1, stride1)

    ref_out = torch.mm(
        utils.to_reference(a, True), utils.to_reference(b.contiguous(), True)
    )
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_unaligned_stride(dtype):
    """#2489 also applies to the ``mm_out`` entry point (shares the guard)."""
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 256, 128, 256
    stride0 = K + 1
    base = torch.randn(M * stride0 + K, dtype=dtype, device=flag_gems.device)
    a = torch.as_strided(base, (M, K), (stride0, 1))
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mm(
        utils.to_reference(a.contiguous(), True), utils.to_reference(b, True)
    )
    with flag_gems.use_gems():
        torch.mm(a, b, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_clean_transpose_unaligned_m(dtype):
    """#2489 for a *clean* transpose whose M is not 16-byte aligned.

    ``torch.mm(x.t(), w)`` (e.g. ``grad_w = x.t() @ grad_out``) makes ``a`` column-major,
    and ``general_mm`` then builds its descriptor from ``a.T.stride() == (M, 1)`` -- so
    the outer stride is M, which ``is_tma_compatible`` never validates (it only checks N
    and K). M = 495 is not a multiple of 8 (fp16/bf16) or 4 (fp32), so the descriptor is
    illegal until the operand is copied.
    """
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 495, 256, 64  # N, K aligned -> TMA is used; M is not
    a = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    assert a.stride() == (1, M) and a.t().is_contiguous()

    ref_out = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_transpose_single_row(dtype):
    """#2489 for an M == 1 operand: ``x.t()`` of a (K, 1) input is (1, K) with stride
    (1, 1), and a size-1 dim never constrains contiguity -- so ``is_contiguous()`` is
    True and both ``.contiguous()`` and ``.clone()`` preserve the stride. The descriptor
    would still get a 1-element (2-byte) outer stride.
    """
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 1, 64, 64
    a = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    assert a.stride() == (1, 1) and a.is_contiguous()

    ref_out = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_column_major(dtype):
    """A column-major ``out``: unlike an operand, ``c`` is fed to the descriptor
    untransposed, so TMA needs its innermost stride to be 1. A transposed ``out`` view
    has stride (1, M) and must be routed through an aligned row-major buffer.
    """
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 64, 64, 64
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((N, M), dtype=dtype, device=flag_gems.device).t()
    assert out.stride() == (1, N)

    ref_out = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    with flag_gems.use_gems():
        torch.mm(a, b, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["unaligned_base", "gapped_stride"])
def test_mm_out_unaligned_output(dtype, layout):
    """#2489 for the *output*: ``general_mm`` builds ``c_desc`` from the caller's ``out``,
    so an unaligned ``out`` view is just as illegal as an unaligned operand. The result
    has to land in the caller's storage, so it is computed into an aligned buffer and
    copied back.
    """
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 64, 64, 64
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)

    if layout == "unaligned_base":
        base = torch.empty(M * N + 16, dtype=dtype, device=flag_gems.device)
        out = torch.as_strided(base, (M, N), (N, 1), storage_offset=1)
        assert out.is_contiguous() and out.data_ptr() % 16 != 0
    else:
        base = torch.empty(M * (N + 1), dtype=dtype, device=flag_gems.device)
        out = torch.as_strided(base, (M, N), (N + 1, 1))
        assert (N + 1) * out.element_size() % 16 != 0

    ref_out = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    with flag_gems.use_gems():
        torch.mm(a, b, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("offset", [1, 2])
def test_mm_unaligned_offset(dtype, offset):
    """#2489 also covers an unaligned *base address*: a stride-contiguous view with
    an odd ``storage_offset`` (e.g. ``weight[:, 1:]``) has 16B-aligned strides but a
    misaligned base, which the host-TMA descriptor rejects ("base must be 16-byte
    aligned"). ``.contiguous()`` is a no-op on such a view, so the fix forces a copy.
    """
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 mm test on tsingmicro platform")
    torch.manual_seed(0)
    M, N, K = 64, 64, 64
    base = torch.randn(M * K + 16, dtype=dtype, device=flag_gems.device)
    a = torch.as_strided(base, (M, K), (K, 1), storage_offset=offset)
    assert a.is_contiguous() and a.data_ptr() % 16 != 0
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)

    ref_out = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    with flag_gems.use_gems():
        res_out = torch.mm(a, b)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
def test_mm_out_vllm_tma_column_major_weight():
    """Regression test for vLLM Inductor mm_out with a column-major BF16 weight."""
    torch.manual_seed(0)
    M, K, N = 4096, 4096, 3328
    dtype = torch.bfloat16

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    mat2 = mat2_storage.t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mat2.shape == (K, N)
    assert mat2.stride() == (1, K)

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_out = torch.empty((M, N), dtype=ref_mat1.dtype, device=ref_mat1.device)
    torch.mm(ref_mat1, ref_mat2, out=ref_out)

    with flag_gems.use_gems():
        torch.mm(mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.skipif(
    not hasattr(
        getattr(getattr(triton, "tools", None), "tensor_descriptor", None),
        "TensorDescriptor",
    ),
    reason="Host TMA TensorDescriptor is required for this regression test.",
)
def test_mm_kernel_general_host_tma_vllm_column_major_weight_compile_error():
    """Reproduce the vLLM TMA descriptor compile error for a column-major BF16 weight."""
    from triton.tools.tensor_descriptor import TensorDescriptor

    from flag_gems.runtime.backend._nvidia.hopper.ops.mm import (
        mm_kernel_general_host_tma,
    )

    torch.manual_seed(0)
    M, K, N = 64, 4096, 3328
    dtype = torch.bfloat16

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2_storage = torch.randn((N, K), dtype=dtype, device=flag_gems.device)
    mat2 = mat2_storage.t()
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)

    assert mat2.shape == (K, N)
    assert mat2.stride() == (1, K)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(mat1, mat1.shape, mat1.stride(), dummy_block)
    b_desc = TensorDescriptor(mat2, mat2.T.shape, mat2.T.stride(), dummy_block)
    c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    mm_kernel_general_host_tma.fn.fn[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=64,
        GROUP_M=8,
        A_ROW_MAJOR=True,
        B_ROW_MAJOR=False,
        dtype="bfloat16",
        num_warps=4,
        num_stages=2,
    )


@pytest.mark.mm
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    ref_mat = utils.to_reference(mat, True)

    ref_out = torch.mm(ref_mat, ref_mat.t())
    with flag_gems.use_gems():
        res_out = torch.mm(mat, mat.t())

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip(
            "Issue #2834: Skipping fp32 mm.out self-transpose test on tsingmicro platform"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, M), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_out = utils.to_reference(out, True)

    torch.mm(ref_mat, ref_mat.t(), out=ref_out)
    with flag_gems.use_gems():
        torch.mm(mat, mat.t(), out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)
