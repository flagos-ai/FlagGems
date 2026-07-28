# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems
from flag_gems.ops.wgrad_gemm_accum import (
    wgrad_gemm_accum_fp16,
    wgrad_gemm_accum_fp32,
    wgrad_gemmex_available,
)

from . import accuracy_utils as utils

try:
    import fused_weight_gradient_mlp_cuda as apex_wgrad

    HAS_APEX_WGRAD = True
except ImportError:
    HAS_APEX_WGRAD = False

# vs-Apex needs a working GemmEx path; Torch fallback is not bit-aligned with Apex.
HAS_WGRAD_GEMMEX = wgrad_gemmex_available()
RUN_VS_APEX_WGRAD = HAS_APEX_WGRAD and HAS_WGRAD_GEMMEX
_SKIP_VS_APEX = "Apex wgrad or FlagGems GemmEx extension unavailable"

WGRAD_SHAPES_2D = [
    (4, 16, 32),
    (8, 32, 64),
    (16, 64, 128),
]

WGRAD_SHAPES_3D = [
    (2, 4, 16, 32),
]

# Leading dims collapse to K = d0 * d1 * d2.
WGRAD_SHAPES_4D = [
    (2, 2, 4, 16, 32),
]

# Training-scale / long-seq 2D shapes — keep identical to
# benchmark/test_wgrad_gemm_accum.py::WGRAD_GEMM_ACCUM_SHAPES.
# Correctness uses vs Apex only (CPU fp64 matmul is too slow at this scale).
WGRAD_SHAPES_LARGE_2D = [
    (64, 512, 1024),
    (128, 1024, 2048),
    (256, 2048, 4096),
    (384, 384, 384),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 4096, 4096),  # large K ≈ token-batch × seq
]

# Collapsed K = dim0 * dim1 matches the large-K 2d case above.
# Keep identical to the last entry of
# benchmark/test_wgrad_gemm_accum.py::WGRAD_GEMM_ACCUM_SHAPES_3D.
WGRAD_SHAPES_LARGE_3D = [
    (8, 1024, 4096, 4096),
]

FP32_ACCUM_INPUT_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    FP32_ACCUM_INPUT_DTYPES.append(torch.bfloat16)

# fp32 activations use cuBLAS tensor-op GEMM (Apex path); CPU fp64 matmul is not
# the right reference on TF32-capable GPUs.  Those cases are covered by vs_apex.
FP32_ACCUM_CPU_REF_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    FP32_ACCUM_CPU_REF_DTYPES.append(torch.bfloat16)
FP32_ACCUM_3D_APEX_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    FP32_ACCUM_3D_APEX_DTYPES.append(torch.bfloat16)

FP16_ACCUM_INPUT_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    FP16_ACCUM_INPUT_DTYPES.append(torch.bfloat16)

# Inner GEMM dimension K = collapsed batch size; scale atol like other BLAS tests.
DEFAULT_ATOL = 1e-4
TF32_OFF_ATOL = 1e-6


def _collapse_to_2d(input_tensor, grad_output):
    if input_tensor.dim() > 2:
        input_2d = input_tensor.reshape(-1, input_tensor.size(-1))
    else:
        input_2d = input_tensor
    if grad_output.dim() > 2:
        grad_output_2d = grad_output.reshape(-1, grad_output.size(-1))
    else:
        grad_output_2d = grad_output
    return input_2d, grad_output_2d


def _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, main_grad):
    """Independent CPU fp64 matmul, accumulated in fp32 (matches main_grad dtype)."""
    ref_input = input_tensor.detach().cpu().double()
    ref_grad_output = grad_output.detach().cpu().double()
    input_2d, grad_output_2d = _collapse_to_2d(ref_input, ref_grad_output)
    wgrad_fp32 = (grad_output_2d.t().contiguous() @ input_2d.contiguous()).float()
    main_grad_fp32 = main_grad.detach().cpu().float().clone()
    main_grad_fp32.add_(wgrad_fp32)
    main_grad.copy_(main_grad_fp32.to(device=main_grad.device, dtype=main_grad.dtype))


def _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, main_grad, dtype):
    """Independent CPU fp64 matmul reference, cast to half storage."""
    ref_input = input_tensor.detach().cpu().double()
    ref_grad_output = grad_output.detach().cpu().double()
    input_2d, grad_output_2d = _collapse_to_2d(ref_input, ref_grad_output)
    wgrad = grad_output_2d.t().contiguous() @ input_2d.contiguous()
    main_grad_cpu = main_grad.detach().cpu().clone()
    main_grad_cpu.add_(wgrad.to(dtype))
    main_grad.copy_(main_grad_cpu)


def _assert_vs_cpu_ref(res, ref, dtype, *, reduce_dim, atol=DEFAULT_ATOL):
    # Always compare on CPU so ``pytest --ref=cpu`` (TO_CPU) is happy:
    # gems_assert_close asserts ``ref`` is already on CPU when TO_CPU is set.
    utils.gems_assert_close(
        res.cpu(),
        ref.cpu(),
        dtype,
        reduce_dim=reduce_dim,
        atol=atol,
    )


def _assert_vs_apex(res, ref, dtype, *, reduce_dim):
    """Apex is the deployment target; compare on CPU for --ref=cpu compatibility."""
    utils.gems_assert_close(
        res.cpu(),
        ref.cpu(),
        dtype,
        reduce_dim=reduce_dim,
        atol=DEFAULT_ATOL,
    )


def _assert_vs_apex_large_k_strict(res, ref, dtype, *, k):
    """Large-K vs Apex without atol*K inflation.

    Default BLAS asserts use ``atol * reduce_dim``; at K~8192 that allows
    ~0.8 absolute error and can hide real drift.  Same-stack backends
    (GemmEx / addmm) should stay close in both max-abs and relative error.
    """
    res_c = res.detach().cpu().float()
    ref_c = ref.detach().cpu().float()
    assert torch.isfinite(res_c).all()
    assert torch.isfinite(ref_c).all()

    diff = (res_c - ref_c).abs()
    max_abs = float(diff.max())
    ref_mag = max(float(ref_c.abs().max()), 1e-6)
    rel_max = max_abs / ref_mag

    # Half storage accum is noisier than fp32 main_grad.
    if dtype in (torch.float16, torch.bfloat16):
        abs_tol, rel_tol = 5e-3, 5e-4
    else:
        abs_tol, rel_tol = 1e-3, 1e-4

    assert max_abs <= abs_tol or rel_max <= rel_tol, (
        f"large-K vs Apex too far: max_abs={max_abs:.6g}, rel_max={rel_max:.6g}, "
        f"k={k}, dtype={dtype}, ref_mag={ref_mag:.6g}"
    )
    torch.testing.assert_close(
        res_c, ref_c, atol=abs_tol, rtol=rel_tol, equal_nan=False
    )


def _assert_boundary_close(
    res, ref, dtype, *, reduce_dim, case, base_atol=DEFAULT_ATOL
):
    """Compare boundary results with magnitude-aware tolerance for large values.

    Unit-scale cases keep the normal BLAS atol/rtol.  For ``large_1e3``, GEMM
    absolute error grows with |A|*|B|*K; default float32 rtol (1.3e-6) is for
    O(1) values and falsely rejects O(1e-5) relative GEMM noise.  Use relative
    comparison at 1e-4 (fp32 matmul across CPU fp64 / GPU / Apex backends).
    """
    res_c = res.detach().cpu().to(dtype)
    ref_c = ref.detach().cpu().to(dtype)
    assert torch.isfinite(res_c).all()
    assert torch.isfinite(ref_c).all()

    if case == "large_1e3":
        mag = max(float(ref_c.abs().max()), 1.0)
        torch.testing.assert_close(
            res_c,
            ref_c,
            atol=base_atol * reduce_dim * mag,
            rtol=1e-4,
            equal_nan=False,
        )
    else:
        utils.gems_assert_close(
            res_c, ref_c, dtype, reduce_dim=reduce_dim, atol=base_atol
        )


def _with_seed(seed: int):
    """Set deterministic seed for reproducible coverage cases."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_with_tf32_disabled(fn):
    """Run function with TF32 disabled, then restore global flags."""
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _as_non_contiguous_2d(contiguous_2d: torch.Tensor) -> torch.Tensor:
    """Build a non-contiguous (B, F) view with identical values."""
    batch, feat = contiguous_2d.shape
    nc = torch.empty(
        feat,
        batch,
        dtype=contiguous_2d.dtype,
        device=contiguous_2d.device,
    ).transpose(0, 1)
    nc.copy_(contiguous_2d)
    assert not nc.is_contiguous()
    assert nc.shape == contiguous_2d.shape
    return nc


def _as_non_contiguous_main_grad(contiguous_2d: torch.Tensor) -> torch.Tensor:
    """Build a non-contiguous (out, in) main_grad with identical values."""
    return _as_non_contiguous_2d(contiguous_2d)


def _as_non_contiguous_3d(contiguous_3d: torch.Tensor) -> torch.Tensor:
    """Build a non-contiguous (D0, D1, F) view with identical values."""
    dim0, dim1, feat = contiguous_3d.shape
    nc = torch.empty(
        dim1,
        dim0,
        feat,
        dtype=contiguous_3d.dtype,
        device=contiguous_3d.device,
    ).transpose(0, 1)
    nc.copy_(contiguous_3d)
    assert not nc.is_contiguous()
    assert nc.shape == contiguous_3d.shape
    return nc


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_2d(batch, in_features, out_features, dtype):
    _with_seed(20260721)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(res_main_grad, ref_main_grad, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_3d(dim0, dim1, in_features, out_features, dtype):
    _with_seed(20260722)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(
        res_main_grad,
        ref_main_grad,
        torch.float32,
        reduce_dim=dim0 * dim1,
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_2d(batch, in_features, out_features, dtype):
    _with_seed(20260723)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main_grad, dtype)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(res_main_grad, ref_main_grad, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_3d(dim0, dim1, in_features, out_features, dtype):
    """fp16/bf16 accum with 3D collapse vs independent CPU fp64 ref."""
    _with_seed(20260753)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    res_main_grad = main_grad.clone()

    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main_grad, dtype)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_main_grad)

    _assert_vs_cpu_ref(res_main_grad, ref_main_grad, dtype, reduce_dim=dim0 * dim1)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_accumulates_twice():
    """Verify += semantics across two micro-batch calls, not overwrite."""
    _with_seed(20260724)
    batch, in_features, out_features = 4, 16, 32
    dtype = torch.float16

    inp1 = torch.randn(batch, in_features, dtype=dtype, device=flag_gems.device)
    gout1 = torch.randn(batch, out_features, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(batch, in_features, dtype=dtype, device=flag_gems.device)
    gout2 = torch.randn(batch, out_features, dtype=dtype, device=flag_gems.device)

    base = torch.zeros(out_features, in_features, dtype=torch.float32, device="cpu")

    ref_main = base.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(inp1, gout1, ref_main)
    _ref_wgrad_gemm_accum_fp32_cpu(inp2, gout2, ref_main)

    res_main = torch.zeros(
        out_features, in_features, dtype=torch.float32, device=flag_gems.device
    )
    wgrad_gemm_accum_fp32(inp1, gout1, res_main)
    wgrad_gemm_accum_fp32(inp2, gout2, res_main)

    _assert_vs_cpu_ref(res_main, ref_main, torch.float32, reduce_dim=2 * batch)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_from_zero_main_grad():
    _with_seed(20260725)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=torch.float16, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=torch.float16, device=flag_gems.device
    )

    ref_main = torch.zeros(out_features, in_features, dtype=torch.float32)
    res_main = torch.zeros(
        out_features, in_features, dtype=torch.float32, device=flag_gems.device
    )

    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main)

    _assert_vs_cpu_ref(res_main, ref_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_accumulates_twice(dtype):
    """fp16/bf16 accum: += across two micro-batches, not overwrite."""
    _with_seed(20260769)
    batch, in_features, out_features = 4, 16, 32

    inp1 = torch.randn(batch, in_features, dtype=dtype, device=flag_gems.device)
    gout1 = torch.randn(batch, out_features, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(batch, in_features, dtype=dtype, device=flag_gems.device)
    gout2 = torch.randn(batch, out_features, dtype=dtype, device=flag_gems.device)

    base = torch.zeros(out_features, in_features, dtype=dtype, device="cpu")

    ref_main = base.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(inp1, gout1, ref_main, dtype)
    _ref_wgrad_gemm_accum_fp16_cpu(inp2, gout2, ref_main, dtype)

    res_main = torch.zeros(
        out_features, in_features, dtype=dtype, device=flag_gems.device
    )
    wgrad_gemm_accum_fp16(inp1, gout1, res_main)
    wgrad_gemm_accum_fp16(inp2, gout2, res_main)

    _assert_vs_cpu_ref(res_main, ref_main, dtype, reduce_dim=2 * batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_from_zero_main_grad(dtype):
    """fp16/bf16 accum starting from zero main_grad."""
    _with_seed(20260770)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )

    ref_main = torch.zeros(out_features, in_features, dtype=dtype)
    res_main = torch.zeros(
        out_features, in_features, dtype=dtype, device=flag_gems.device
    )

    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main, dtype)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_main)

    _assert_vs_cpu_ref(res_main, ref_main, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_main_grad_shape():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    # Expected main_grad shape is (32, 16); use transposed (16, 32).
    main_grad = torch.zeros(16, 32, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad shape mismatch"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_main_grad_dtype():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad must be float32"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_grad_output_dtype():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float32, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="grad_output dtype must match input dtype"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_row_mismatch():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(5, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="same number of rows after collapse"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
def test_wgrad_gemm_accum_fp32_invalid_input_dtype():
    input_tensor = torch.randn(4, 16, dtype=torch.float64, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float64, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="Unsupported input dtype"):
        wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp16
def test_wgrad_gemm_accum_fp16_invalid_main_grad_shape():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(16, 32, dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad shape mismatch"):
        wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp16
def test_wgrad_gemm_accum_fp16_invalid_main_grad_dtype():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad must be float16 or bfloat16"):
        wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not utils.bf16_is_supported,
    reason="bfloat16 not supported on this device",
)
def test_wgrad_gemm_accum_fp16_invalid_main_grad_dtype_mismatch():
    """fp16 activations with bf16 main_grad must be rejected."""
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.bfloat16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="main_grad dtype must match input dtype"):
        wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp16
def test_wgrad_gemm_accum_fp16_invalid_grad_output_dtype():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(4, 32, dtype=torch.float32, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="grad_output dtype must match input dtype"):
        wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp16
def test_wgrad_gemm_accum_fp16_invalid_row_mismatch():
    input_tensor = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    grad_output = torch.randn(5, 32, dtype=torch.float16, device=flag_gems.device)
    main_grad = torch.zeros(32, 16, dtype=torch.float16, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="same number of rows after collapse"):
        wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_empty_batch(dtype):
    """K==0 must be a no-op: main_grad unchanged."""
    _with_seed(20260739)
    batch, in_features, out_features = 0, 16, 32
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )
    seed = main_grad.clone()

    wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)
    assert torch.equal(main_grad, seed)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_empty_batch(dtype):
    """K==0 must be a no-op on the fp16/bf16 accum path."""
    _with_seed(20260740)
    batch, in_features, out_features = 0, 16, 32
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    seed = main_grad.clone()

    wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)
    assert torch.equal(main_grad, seed)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_zero_in_features(dtype):
    """in_features==0: empty product, main_grad (out, 0) unchanged."""
    _with_seed(20260761)
    batch, in_features, out_features = 8, 0, 32
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )
    seed = main_grad.clone()

    wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)
    assert main_grad.shape == (out_features, 0)
    assert torch.equal(main_grad, seed)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_zero_out_features(dtype):
    """out_features==0: empty product, main_grad (0, in) unchanged."""
    _with_seed(20260762)
    batch, in_features, out_features = 8, 16, 0
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )
    seed = main_grad.clone()

    wgrad_gemm_accum_fp32(input_tensor, grad_output, main_grad)
    assert main_grad.shape == (0, in_features)
    assert torch.equal(main_grad, seed)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_zero_in_features(dtype):
    """fp16/bf16 accum: in_features==0 is a no-op."""
    _with_seed(20260763)
    batch, in_features, out_features = 8, 0, 32
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    seed = main_grad.clone()

    wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)
    assert torch.equal(main_grad, seed)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_zero_out_features(dtype):
    """fp16/bf16 accum: out_features==0 is a no-op."""
    _with_seed(20260764)
    batch, in_features, out_features = 8, 16, 0
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    seed = main_grad.clone()

    wgrad_gemm_accum_fp16(input_tensor, grad_output, main_grad)
    assert torch.equal(main_grad, seed)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("zero_dim", ["in", "out"])
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_zero_features(zero_dim, dtype):
    """Apex/cublasGemmEx rejects zero M/N; gems treats empty product as no-op."""
    _with_seed(20260765)
    batch = 8
    in_features = 0 if zero_dim == "in" else 16
    out_features = 0 if zero_dim == "out" else 32
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main = main_grad_seed.clone()
    with pytest.raises(RuntimeError):
        apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)

    gems_main = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)
    assert torch.equal(gems_main, main_grad_seed)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("zero_dim", ["in", "out"])
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_zero_features(zero_dim, dtype):
    """Same divergence on fp16/bf16 accum: Apex errors, gems no-op."""
    _with_seed(20260766)
    batch = 8
    in_features = 0 if zero_dim == "in" else 16
    out_features = 0 if zero_dim == "out" else 32
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main = main_grad_seed.clone()
    with pytest.raises(RuntimeError):
        apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main)

    gems_main = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main)
    assert torch.equal(gems_main, main_grad_seed)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_main_grad_non_contiguous(dtype):
    """Non-contiguous main_grad must match contiguous accumulation."""
    _with_seed(20260741)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_c = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main = main_c.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)

    res_contig = main_c.clone()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_contig)

    # Rebuild non-contiguous storage; Tensor.clone() would densify and defeat the test.
    res_nc = _as_non_contiguous_main_grad(main_c)
    assert not res_nc.is_contiguous()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, torch.float32, reduce_dim=batch)
    _assert_vs_cpu_ref(res_nc, res_contig, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_main_grad_non_contiguous(dtype):
    """Non-contiguous main_grad on fp16/bf16 accum path."""
    _with_seed(20260742)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_c = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_main = main_c.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main, dtype)

    res_contig = main_c.clone()
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_contig)

    res_nc = _as_non_contiguous_main_grad(main_c)
    assert not res_nc.is_contiguous()
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, dtype, reduce_dim=batch)
    _assert_vs_cpu_ref(res_nc, res_contig, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex(batch, in_features, out_features, dtype):
    _with_seed(20260726)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(gems_main_grad, apex_main_grad, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_3D_APEX_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_3d(dim0, dim1, in_features, out_features, dtype):
    _with_seed(20260727)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(
        gems_main_grad, apex_main_grad, torch.float32, reduce_dim=dim0 * dim1
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dim0, dim1, dim2, in_features, out_features", WGRAD_SHAPES_4D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_3D_APEX_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_4d(
    dim0, dim1, dim2, in_features, out_features, dtype
):
    """4D activations collapse to 2D (K = d0*d1*d2) must match Apex."""
    _with_seed(20260767)
    input_tensor = torch.randn(
        (dim0, dim1, dim2, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, dim2, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(
        gems_main_grad,
        apex_main_grad,
        torch.float32,
        reduce_dim=dim0 * dim1 * dim2,
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex(batch, in_features, out_features, dtype):
    _with_seed(20260728)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(gems_main_grad, apex_main_grad, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_LARGE_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_large_shape(
    batch, in_features, out_features, dtype
):
    """Large / long-seq shapes from the benchmark suite, vs Apex."""
    _with_seed(20260745)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex_large_k_strict(
        gems_main_grad, apex_main_grad, torch.float32, k=batch
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_LARGE_2D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_large_shape(
    batch, in_features, out_features, dtype
):
    """Large / long-seq shapes on fp16/bf16 accum path, vs Apex."""
    _with_seed(20260746)
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex_large_k_strict(gems_main_grad, apex_main_grad, dtype, k=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_LARGE_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_3D_APEX_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_large_shape_3d(
    dim0, dim1, in_features, out_features, dtype
):
    """3D long-seq collapse (large K) vs Apex on fp32 accum."""
    _with_seed(20260747)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex_large_k_strict(
        gems_main_grad, apex_main_grad, torch.float32, k=dim0 * dim1
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_3d(dim0, dim1, in_features, out_features, dtype):
    """fp16/bf16 accum 3D collapse must match Apex."""
    _with_seed(20260754)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(gems_main_grad, apex_main_grad, dtype, reduce_dim=dim0 * dim1)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dim0, dim1, dim2, in_features, out_features", WGRAD_SHAPES_4D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_4d(
    dim0, dim1, dim2, in_features, out_features, dtype
):
    """fp16/bf16 accum 4D collapse must match Apex."""
    _with_seed(20260768)
    input_tensor = torch.randn(
        (dim0, dim1, dim2, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, dim2, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex(
        gems_main_grad, apex_main_grad, dtype, reduce_dim=dim0 * dim1 * dim2
    )


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_LARGE_3D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_large_shape_3d(
    dim0, dim1, in_features, out_features, dtype
):
    """Large 3D long-seq collapse on fp16/bf16 accum vs Apex."""
    _with_seed(20260755)
    input_tensor = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main_grad = main_grad_seed.clone()
    gems_main_grad = main_grad_seed.clone()

    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main_grad)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main_grad)

    _assert_vs_apex_large_k_strict(gems_main_grad, apex_main_grad, dtype, k=dim0 * dim1)


# Repeat-call stability: catch intermittent handle / workspace pollution.
REPEAT_ITERS_FRESH = 200
REPEAT_ITERS_ACCUM = 200
REPEAT_ITERS_STRESS = 1000


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_repeat_fresh(dtype):
    """Same inputs, many single-shot calls; each must still match Apex."""
    _with_seed(20260748)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    for _ in range(REPEAT_ITERS_FRESH):
        apex_main = main_grad_seed.clone()
        gems_main = main_grad_seed.clone()
        apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)
        wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)
        _assert_vs_apex(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_repeat_fresh(dtype):
    """Same inputs, many single-shot calls on fp16/bf16 accum path."""
    _with_seed(20260749)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    for _ in range(REPEAT_ITERS_FRESH):
        apex_main = main_grad_seed.clone()
        gems_main = main_grad_seed.clone()
        apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main)
        wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main)
        _assert_vs_apex(gems_main, apex_main, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_repeat_accum(dtype):
    """Accumulate repeatedly into the same main_grad; final must match Apex."""
    _with_seed(20260750)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    apex_main = main_grad_seed.clone()
    gems_main = main_grad_seed.clone()
    for _ in range(REPEAT_ITERS_ACCUM):
        apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)
        wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)

    _assert_vs_apex(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_repeat_accum(dtype):
    """Accumulate repeatedly on fp16/bf16 path; final must match Apex."""
    _with_seed(20260751)
    batch, in_features, out_features = 8, 32, 64
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    apex_main = main_grad_seed.clone()
    gems_main = main_grad_seed.clone()
    for _ in range(REPEAT_ITERS_ACCUM):
        apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main)
        wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main)

    _assert_vs_apex(gems_main, apex_main, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
def test_wgrad_gemm_accum_fp32_vs_apex_repeat_stress():
    """1000 single-shot calls on the common fp16→fp32 training path."""
    _with_seed(20260752)
    batch, in_features, out_features = 8, 32, 64
    dtype = torch.float16
    input_tensor = torch.randn(
        (batch, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    for _ in range(REPEAT_ITERS_STRESS):
        apex_main = main_grad_seed.clone()
        gems_main = main_grad_seed.clone()
        apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)
        wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)
        _assert_vs_apex(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.wgrad_fp32_tf32_off_strict
@pytest.mark.parametrize(
    "batch, in_features, out_features",
    [
        (4, 3072, 4096),  # small batch, large hidden
        (257, 129, 257),  # non-aligned dimensions
        (1024, 64, 64),  # large K accumulation
    ],
)
def test_wgrad_gemm_accum_fp32_cpu_ref_strict_with_tf32_off(
    batch, in_features, out_features
):
    """Mathematical strictness check for fp32 inputs under full-fp32 GEMM."""
    _with_seed(20260729)
    input_tensor = torch.randn(
        (batch, in_features), dtype=torch.float32, device=flag_gems.device
    )
    grad_output = torch.randn(
        (batch, out_features), dtype=torch.float32, device=flag_gems.device
    )
    main_grad = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    ref_main_grad = main_grad.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main_grad)

    res_main_grad = main_grad.clone()
    _run_with_tf32_disabled(
        lambda: wgrad_gemm_accum_fp32(
            input_tensor, grad_output, res_main_grad, strict_cpu_ref=True
        )
    )

    utils.gems_assert_close(
        res_main_grad.cpu(),
        ref_main_grad.cpu(),
        torch.float32,
        reduce_dim=batch,
        atol=TF32_OFF_ATOL,
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
@pytest.mark.parametrize("layout", ["input_nc", "grad_output_nc", "both_nc"])
def test_wgrad_gemm_accum_fp32_2d_non_contiguous(
    batch, in_features, out_features, dtype, layout
):
    """Non-contiguous 2D inputs must match contiguous results and CPU ref."""
    _with_seed(20260730)
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_tensor = input_c
    grad_output = grad_output_c
    if layout in ("input_nc", "both_nc"):
        input_tensor = _as_non_contiguous_2d(input_c)
    if layout in ("grad_output_nc", "both_nc"):
        grad_output = _as_non_contiguous_2d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, torch.float32, reduce_dim=batch)
    _assert_vs_cpu_ref(res_nc, res_contig, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_3d_non_contiguous(
    dim0, dim1, in_features, out_features, dtype
):
    """Non-contiguous 3D inputs must match contiguous results and CPU ref."""
    _with_seed(20260731)
    input_c = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output_c = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_tensor = _as_non_contiguous_3d(input_c)
    grad_output = _as_non_contiguous_3d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, torch.float32, reduce_dim=dim0 * dim1)
    _assert_vs_cpu_ref(res_nc, res_contig, torch.float32, reduce_dim=dim0 * dim1)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D[:1])
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
@pytest.mark.parametrize("layout", ["input_nc", "grad_output_nc", "both_nc"])
def test_wgrad_gemm_accum_fp32_vs_apex_non_contiguous(
    batch, in_features, out_features, dtype, layout
):
    """Gems on non-contiguous views must match Apex on contiguous equivalents."""
    _with_seed(20260732)
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_tensor = input_c
    grad_output = grad_output_c
    if layout in ("input_nc", "both_nc"):
        input_tensor = _as_non_contiguous_2d(input_c)
    if layout in ("grad_output_nc", "both_nc"):
        grad_output = _as_non_contiguous_2d(grad_output_c)

    apex_main = main_grad_seed.clone()
    gems_main = main_grad_seed.clone()

    # Apex uses tensor.view in its CUDA stub; feed contiguous equivalents.
    apex_wgrad.wgrad_gemm_accum_fp32(input_c, grad_output_c, apex_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)

    _assert_vs_apex(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("batch, in_features, out_features", WGRAD_SHAPES_2D[:1])
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
@pytest.mark.parametrize("layout", ["input_nc", "grad_output_nc", "both_nc"])
def test_wgrad_gemm_accum_fp16_2d_non_contiguous(
    batch, in_features, out_features, dtype, layout
):
    """fp16 accum path: non-contiguous inputs match contiguous and CPU ref."""
    _with_seed(20260733)
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    input_tensor = input_c
    grad_output = grad_output_c
    if layout in ("input_nc", "both_nc"):
        input_tensor = _as_non_contiguous_2d(input_c)
    if layout in ("grad_output_nc", "both_nc"):
        grad_output = _as_non_contiguous_2d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main, dtype)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, dtype, reduce_dim=batch)
    _assert_vs_cpu_ref(res_nc, res_contig, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dim0, dim1, in_features, out_features", WGRAD_SHAPES_3D)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_3d_non_contiguous(
    dim0, dim1, in_features, out_features, dtype
):
    """fp16/bf16 accum: non-contiguous 3D inputs match contiguous and CPU ref."""
    _with_seed(20260756)
    input_c = torch.randn(
        (dim0, dim1, in_features), dtype=dtype, device=flag_gems.device
    )
    grad_output_c = torch.randn(
        (dim0, dim1, out_features), dtype=dtype, device=flag_gems.device
    )
    main_grad_seed = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    input_tensor = _as_non_contiguous_3d(input_c)
    grad_output = _as_non_contiguous_3d(grad_output_c)

    ref_main = main_grad_seed.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main, dtype)

    res_contig = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_c, grad_output_c, res_contig)

    res_nc = main_grad_seed.clone()
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_nc)

    _assert_vs_cpu_ref(res_nc, ref_main, dtype, reduce_dim=dim0 * dim1)
    _assert_vs_cpu_ref(res_nc, res_contig, dtype, reduce_dim=dim0 * dim1)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_multi_non_contiguous(dtype):
    """input + grad_output + main_grad all non-contiguous vs contiguous path."""
    _with_seed(20260757)
    batch, in_features, out_features = 8, 32, 64
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_c = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_nc = _as_non_contiguous_2d(input_c)
    grad_output_nc = _as_non_contiguous_2d(grad_output_c)
    main_nc = _as_non_contiguous_main_grad(main_c)
    assert not input_nc.is_contiguous()
    assert not grad_output_nc.is_contiguous()
    assert not main_nc.is_contiguous()

    ref_main = main_c.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_c, grad_output_c, ref_main)

    res_contig = main_c.clone()
    wgrad_gemm_accum_fp32(input_c, grad_output_c, res_contig)

    wgrad_gemm_accum_fp32(input_nc, grad_output_nc, main_nc)

    _assert_vs_cpu_ref(main_nc, ref_main, torch.float32, reduce_dim=batch)
    _assert_vs_cpu_ref(main_nc, res_contig, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_multi_non_contiguous(dtype):
    """fp16/bf16: input + grad_output + main_grad all non-contiguous."""
    _with_seed(20260758)
    batch, in_features, out_features = 8, 32, 64
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_c = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    input_nc = _as_non_contiguous_2d(input_c)
    grad_output_nc = _as_non_contiguous_2d(grad_output_c)
    main_nc = _as_non_contiguous_main_grad(main_c)
    assert not main_nc.is_contiguous()

    ref_main = main_c.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(input_c, grad_output_c, ref_main, dtype)

    res_contig = main_c.clone()
    wgrad_gemm_accum_fp16(input_c, grad_output_c, res_contig)

    wgrad_gemm_accum_fp16(input_nc, grad_output_nc, main_nc)

    _assert_vs_cpu_ref(main_nc, ref_main, dtype, reduce_dim=batch)
    _assert_vs_cpu_ref(main_nc, res_contig, dtype, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_multi_non_contiguous(dtype):
    """All-non-contiguous gems path must match Apex on contiguous tensors."""
    _with_seed(20260759)
    batch, in_features, out_features = 8, 32, 64
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_c = torch.randn(
        (out_features, in_features), dtype=torch.float32, device=flag_gems.device
    )

    input_nc = _as_non_contiguous_2d(input_c)
    grad_output_nc = _as_non_contiguous_2d(grad_output_c)
    gems_main = _as_non_contiguous_main_grad(main_c)

    apex_main = main_c.clone()
    apex_wgrad.wgrad_gemm_accum_fp32(input_c, grad_output_c, apex_main)
    wgrad_gemm_accum_fp32(input_nc, grad_output_nc, gems_main)

    _assert_vs_apex(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_multi_non_contiguous(dtype):
    """fp16/bf16 all-non-contiguous path vs Apex contiguous."""
    _with_seed(20260760)
    batch, in_features, out_features = 8, 32, 64
    input_c = torch.randn((batch, in_features), dtype=dtype, device=flag_gems.device)
    grad_output_c = torch.randn(
        (batch, out_features), dtype=dtype, device=flag_gems.device
    )
    main_c = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    input_nc = _as_non_contiguous_2d(input_c)
    grad_output_nc = _as_non_contiguous_2d(grad_output_c)
    gems_main = _as_non_contiguous_main_grad(main_c)

    apex_main = main_c.clone()
    apex_wgrad.wgrad_gemm_accum_fp16(input_c, grad_output_c, apex_main)
    wgrad_gemm_accum_fp16(input_nc, grad_output_nc, gems_main)

    _assert_vs_apex(gems_main, apex_main, dtype, reduce_dim=batch)


def _make_numeric_boundary_tensors(
    case, *, batch, in_features, out_features, dtype, device, seed
):
    """Build input / grad_output / main_grad for numeric boundary cases."""
    _with_seed(seed)
    if case == "zeros":
        input_tensor = torch.zeros((batch, in_features), dtype=dtype, device=device)
        grad_output = torch.zeros((batch, out_features), dtype=dtype, device=device)
        # Non-zero main_grad: zero GEMM must leave it unchanged.
        main_grad = torch.randn(
            (out_features, in_features), dtype=torch.float32, device=device
        )
    elif case == "large_1e3":
        scale = 1e3
        input_tensor = (
            torch.randn((batch, in_features), dtype=dtype, device=device) * scale
        )
        grad_output = (
            torch.randn((batch, out_features), dtype=dtype, device=device) * scale
        )
        main_grad = (
            torch.randn((out_features, in_features), dtype=torch.float32, device=device)
            * scale
        )
    elif case == "small_1e-5":
        scale = 1e-5
        input_tensor = (
            torch.randn((batch, in_features), dtype=dtype, device=device) * scale
        )
        grad_output = (
            torch.randn((batch, out_features), dtype=dtype, device=device) * scale
        )
        main_grad = (
            torch.randn((out_features, in_features), dtype=torch.float32, device=device)
            * scale
        )
    elif case == "mixed_signs":
        input_tensor = torch.randn((batch, in_features), dtype=dtype, device=device)
        grad_output = torch.randn((batch, out_features), dtype=dtype, device=device)
        main_grad = torch.randn(
            (out_features, in_features), dtype=torch.float32, device=device
        )
        input_tensor[: max(batch // 2, 1)].neg_()
        grad_output[max(batch // 2, 1) :].neg_()
        main_grad[: max(out_features // 2, 1)].neg_()
    else:
        raise ValueError(f"Unknown numeric boundary case: {case}")

    return input_tensor, grad_output, main_grad


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.wgrad_fp32_tf32_off_strict
@pytest.mark.parametrize(
    "case",
    ["zeros", "large_1e3", "small_1e-5", "mixed_signs"],
)
@pytest.mark.parametrize("dtype", FP32_ACCUM_CPU_REF_DTYPES)
def test_wgrad_gemm_accum_fp32_numeric_boundaries(case, dtype):
    """Cover zeros / large / small / mixed signs vs independent CPU fp64 ref."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_numeric_boundary_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        device=flag_gems.device,
        seed=20260734,
    )

    ref_main = main_grad.clone()
    res_main = main_grad.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, res_main)

    assert torch.isfinite(res_main).all()
    assert torch.isfinite(ref_main).all()

    if case == "zeros":
        # Exact no-op: GEMM contribution is all zeros.
        assert torch.equal(res_main, main_grad)
        assert torch.equal(ref_main, main_grad)
    else:
        _assert_boundary_close(
            res_main, ref_main, torch.float32, reduce_dim=batch, case=case
        )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.parametrize(
    "case",
    ["zeros", "large_1e3", "small_1e-5", "mixed_signs"],
)
def test_wgrad_gemm_accum_fp32_numeric_boundaries_fp32_input_tf32_off(case):
    """fp32 activations: math check under full fp32 GEMM (TF32 disabled)."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_numeric_boundary_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        dtype=torch.float32,
        device=flag_gems.device,
        seed=20260735,
    )

    ref_main = main_grad.clone()
    _ref_wgrad_gemm_accum_fp32_cpu(input_tensor, grad_output, ref_main)

    res_main = main_grad.clone()
    _run_with_tf32_disabled(
        lambda: wgrad_gemm_accum_fp32(
            input_tensor, grad_output, res_main, strict_cpu_ref=True
        )
    )

    assert torch.isfinite(res_main).all()
    assert torch.isfinite(ref_main).all()

    if case == "zeros":
        assert torch.equal(res_main, main_grad)
        assert torch.equal(ref_main, main_grad)
    else:
        _assert_boundary_close(
            res_main,
            ref_main,
            torch.float32,
            reduce_dim=batch,
            case=case,
            base_atol=TF32_OFF_ATOL,
        )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize(
    "case",
    ["zeros", "large_1e3", "small_1e-5", "mixed_signs"],
)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_numeric_boundaries(case, dtype):
    """Boundary values must also match Apex on the same tensors."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_numeric_boundary_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        device=flag_gems.device,
        seed=20260736,
    )

    apex_main = main_grad.clone()
    gems_main = main_grad.clone()
    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)

    assert torch.isfinite(gems_main).all()
    assert torch.isfinite(apex_main).all()

    if case == "zeros":
        assert torch.equal(gems_main, main_grad)
        assert torch.equal(apex_main, main_grad)
    else:
        _assert_boundary_close(
            gems_main, apex_main, torch.float32, reduce_dim=batch, case=case
        )


def _large_activation_scale(dtype):
    """Pick a large magnitude that stays finite for the given storage dtype.

    For fp16 accum, GEMM of O(1e3)*O(1e3) over K~8 overflows fp16 (~6.5e4).
    Keep the intent (large vs unit randn) without turning the test into Inf-only.
    """
    if dtype == torch.float16:
        return 64.0
    return 1e3


def _make_fp16_accum_boundary_tensors(
    case, *, batch, in_features, out_features, dtype, device, seed
):
    """Build tensors for fp16/bf16 accum numeric boundary cases."""
    _with_seed(seed)
    if case == "zeros":
        input_tensor = torch.zeros((batch, in_features), dtype=dtype, device=device)
        grad_output = torch.zeros((batch, out_features), dtype=dtype, device=device)
        main_grad = torch.randn((out_features, in_features), dtype=dtype, device=device)
    elif case == "large_1e3":
        scale = _large_activation_scale(dtype)
        input_tensor = (
            torch.randn((batch, in_features), dtype=dtype, device=device) * scale
        )
        grad_output = (
            torch.randn((batch, out_features), dtype=dtype, device=device) * scale
        )
        main_grad = (
            torch.randn((out_features, in_features), dtype=dtype, device=device) * scale
        )
    elif case == "small_1e-5":
        scale = 1e-5
        input_tensor = (
            torch.randn((batch, in_features), dtype=dtype, device=device) * scale
        )
        grad_output = (
            torch.randn((batch, out_features), dtype=dtype, device=device) * scale
        )
        main_grad = (
            torch.randn((out_features, in_features), dtype=dtype, device=device) * scale
        )
    elif case == "mixed_signs":
        input_tensor = torch.randn((batch, in_features), dtype=dtype, device=device)
        grad_output = torch.randn((batch, out_features), dtype=dtype, device=device)
        main_grad = torch.randn((out_features, in_features), dtype=dtype, device=device)
        input_tensor[: max(batch // 2, 1)].neg_()
        grad_output[max(batch // 2, 1) :].neg_()
        main_grad[: max(out_features // 2, 1)].neg_()
    else:
        raise ValueError(f"Unknown numeric boundary case: {case}")

    return input_tensor, grad_output, main_grad


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.parametrize(
    "case",
    ["zeros", "large_1e3", "small_1e-5", "mixed_signs"],
)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_numeric_boundaries(case, dtype):
    """fp16/bf16 accum path: same boundary coverage vs CPU fp64 ref."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_fp16_accum_boundary_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        device=flag_gems.device,
        seed=20260737,
    )

    ref_main = main_grad.clone()
    res_main = main_grad.clone()
    _ref_wgrad_gemm_accum_fp16_cpu(input_tensor, grad_output, ref_main, dtype)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, res_main)

    assert torch.isfinite(res_main).all()
    assert torch.isfinite(ref_main).all()

    if case == "zeros":
        assert torch.equal(res_main, main_grad)
        assert torch.equal(ref_main, main_grad)
    else:
        _assert_boundary_close(res_main, ref_main, dtype, reduce_dim=batch, case=case)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize(
    "case",
    ["zeros", "large_1e3", "small_1e-5", "mixed_signs"],
)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_numeric_boundaries(case, dtype):
    """fp16/bf16 accum boundary values must also match Apex on the same tensors."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_fp16_accum_boundary_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        dtype=dtype,
        device=flag_gems.device,
        seed=20260738,
    )

    apex_main = main_grad.clone()
    gems_main = main_grad.clone()
    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main)

    assert torch.isfinite(gems_main).all()
    assert torch.isfinite(apex_main).all()

    if case == "zeros":
        assert torch.equal(gems_main, main_grad)
        assert torch.equal(apex_main, main_grad)
    else:
        _assert_boundary_close(gems_main, apex_main, dtype, reduce_dim=batch, case=case)


def _make_nan_inf_tensors(
    case,
    *,
    batch,
    in_features,
    out_features,
    input_dtype,
    main_dtype,
    device,
    seed,
):
    """Build tensors with a single NaN/Inf injected for propagation checks."""
    _with_seed(seed)
    input_tensor = torch.randn((batch, in_features), dtype=input_dtype, device=device)
    grad_output = torch.randn((batch, out_features), dtype=input_dtype, device=device)
    main_grad = torch.randn(
        (out_features, in_features), dtype=main_dtype, device=device
    )

    if case == "nan_in_input":
        input_tensor[0, 0] = float("nan")
    elif case == "nan_in_grad_output":
        grad_output[0, 0] = float("nan")
    elif case == "inf_in_input":
        input_tensor[0, 0] = float("inf")
    elif case == "inf_in_grad_output":
        grad_output[0, 0] = float("inf")
    elif case == "neg_inf_in_input":
        input_tensor[0, 0] = float("-inf")
    elif case == "nan_in_main_grad":
        main_grad[0, 0] = float("nan")
    elif case == "inf_in_main_grad":
        main_grad[0, 0] = float("inf")
    else:
        raise ValueError(f"unknown nan/inf case: {case}")
    return input_tensor, grad_output, main_grad


_NAN_INF_CASES = [
    "nan_in_input",
    "nan_in_grad_output",
    "inf_in_input",
    "inf_in_grad_output",
    "neg_inf_in_input",
    "nan_in_main_grad",
    "inf_in_main_grad",
]


def _assert_vs_apex_equal_nan(res, ref, dtype, *, reduce_dim):
    """Match Apex including NaN/Inf positions (GEMM propagation semantics)."""
    utils.gems_assert_close(
        res.cpu(),
        ref.cpu(),
        dtype,
        equal_nan=True,
        reduce_dim=reduce_dim,
        atol=DEFAULT_ATOL,
    )


@pytest.mark.wgrad_gemm_accum_fp32
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("case", _NAN_INF_CASES)
@pytest.mark.parametrize("dtype", FP32_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp32_vs_apex_nan_inf(case, dtype):
    """NaN/Inf must propagate the same way as Apex on the fp32-accum path."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_nan_inf_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        input_dtype=dtype,
        main_dtype=torch.float32,
        device=flag_gems.device,
        seed=20260743,
    )

    apex_main = main_grad.clone()
    gems_main = main_grad.clone()
    apex_wgrad.wgrad_gemm_accum_fp32(input_tensor, grad_output, apex_main)
    wgrad_gemm_accum_fp32(input_tensor, grad_output, gems_main)

    _assert_vs_apex_equal_nan(gems_main, apex_main, torch.float32, reduce_dim=batch)


@pytest.mark.wgrad_gemm_accum_fp16
@pytest.mark.skipif(
    not RUN_VS_APEX_WGRAD,
    reason=_SKIP_VS_APEX,
)
@pytest.mark.parametrize("case", _NAN_INF_CASES)
@pytest.mark.parametrize("dtype", FP16_ACCUM_INPUT_DTYPES)
def test_wgrad_gemm_accum_fp16_vs_apex_nan_inf(case, dtype):
    """NaN/Inf must propagate the same way as Apex on the fp16/bf16 accum path."""
    batch, in_features, out_features = 8, 32, 64
    input_tensor, grad_output, main_grad = _make_nan_inf_tensors(
        case,
        batch=batch,
        in_features=in_features,
        out_features=out_features,
        input_dtype=dtype,
        main_dtype=dtype,
        device=flag_gems.device,
        seed=20260744,
    )

    apex_main = main_grad.clone()
    gems_main = main_grad.clone()
    apex_wgrad.wgrad_gemm_accum_fp16(input_tensor, grad_output, apex_main)
    wgrad_gemm_accum_fp16(input_tensor, grad_output, gems_main)

    _assert_vs_apex_equal_nan(gems_main, apex_main, dtype, reduce_dim=batch)
