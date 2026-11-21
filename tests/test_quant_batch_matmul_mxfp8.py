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

"""Tests for the Ascend MX-FP8 quant_batch_matmul_mxfp8 operator."""

import importlib

import pytest
import torch

import flag_gems

from . import conftest as cfg

try:
    import torch_npu
except ImportError:
    torch_npu = None

ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="quant_batch_matmul_mxfp8 targets Ascend 950",
)

pytestmark = [pytest.mark.quant_batch_matmul_mxfp8, ascend_only]

MX = 32
ACL_FORMAT_FRACTAL_NZ = 29

# ----- model shapes from profiling (model, M, K, N, nz, weight_pct) -----
MODEL_SHAPES_FULL = (
    # name,        M,  K,     N,      nz
    ("GLM-W4",     1,  2048,  16384,  True),
    ("GLM-W8",     1,  2048,  16384,  True),
    ("DSV4",       1,  1024,  32768,  False),
    ("M3",         8,  2048,  6144,   True),
    ("Qwen",       1,  2048,  512,    False),
)

MODEL_SHAPES = (
    (MODEL_SHAPES_FULL[0], MODEL_SHAPES_FULL[2], MODEL_SHAPES_FULL[4])
    if cfg.QUICK_MODE
    else MODEL_SHAPES_FULL
)

# ----- boundary / edge cases -----
BOUNDARY_SHAPES_FULL = (
    # name,          M,  K,    N,    nz
    ("K=64",         1,  64,   1024, False),   # minimum K
    ("K=128",        1,  128,  1024, False),   # BK boundary
    ("small-N-ND",   1,  2048, 64,   False),   # small N
    ("M=2-ND",       2,  2048, 1024, False),   # M_CE path (1 < M < 16)
    ("M=4-NZ",       4,  2048, 1024, True),    # M_CE path with NZ
    ("M=15-ND",      15, 2048, 1024, False),   # M_CE upper edge
    ("M=16-NZ",      16, 2048, 1024, True),    # exact tile, no padding
    ("M=32-ND",      32, 2048, 1024, False),   # multi m-tile, no split-K
)

BOUNDARY_SHAPES = (
    (BOUNDARY_SHAPES_FULL[0], BOUNDARY_SHAPES_FULL[3], BOUNDARY_SHAPES_FULL[6])
    if cfg.QUICK_MODE
    else BOUNDARY_SHAPES_FULL
)


# ---------------------------------------------------------------------- helpers


def _build_mxfp8_inputs(M, K, N, nz=True, seed=0):
    """Build MXFP8 inputs with npu_dynamic_mx_quant for realistic e8m0 scales.

    Layout matches vllm-ascend ``AscendW8A8MXFP8LinearMethod.process_weights_after_loading``:
      weight:       [N, K] -> transpose -> [K, N], optionally NZ
      weight_scale: [N, K/32] -> [N, K/64, 2] -> transpose(0,1) -> [K/64, N, 2]
      pertoken_scale: from npu_dynamic_mx_quant, shape [M, K/64, 2]
    """
    dev = flag_gems.device
    torch.manual_seed(seed)
    x_bf = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w_bf = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.1

    x, sa = torch_npu.npu_dynamic_mx_quant(x_bf, dst_type=torch.float8_e4m3fn)
    wq, sq = torch_npu.npu_dynamic_mx_quant(w_bf, dst_type=torch.float8_e4m3fn)

    w_nd = wq.t().contiguous()
    sb = sq.reshape(N, K // 64, 2).transpose(0, 1).contiguous()

    w = w_nd
    if nz:
        w = torch_npu.npu_format_cast(
            w_nd, ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
        )
    return x, sa.contiguous(), w, w_nd, sb


def _cann_baseline(x, sa, w, sb, output_dtype=torch.bfloat16):
    """CANN baseline: torch_npu.npu_quant_matmul in the MXFP8 form."""
    return torch_npu.npu_quant_matmul(
        x, w, sb,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=sa,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=None,
        output_dtype=output_dtype,
        group_sizes=[1, 1, MX],
    )


def _torch_reference(x, w_nd, sb, sa, output_dtype=torch.bfloat16):
    """Pure torch fp32 reference for cross-validation (ND weight only)."""
    M, K = x.shape
    _, N = w_nd.shape
    nb = K // MX

    def e8m0(v):
        vi = v.to(torch.int32)
        return torch.where(
            vi == 0,
            torch.zeros_like(vi, dtype=torch.float32),
            torch.exp2((vi - 127).to(torch.float32)),
        )

    sa_flat = sa.reshape(-1)[:M * nb].reshape(M, nb)
    sb_flat = sb.reshape(-1)
    sb_2d = sb_flat.reshape(nb // 2, N, 2).permute(0, 2, 1).reshape(nb, N)

    xd = x.float() * e8m0(sa_flat).repeat_interleave(MX, dim=-1)
    wd = w_nd.float() * e8m0(sb_2d).repeat_interleave(MX, dim=0)
    return (xd @ wd).to(output_dtype)


# ------------------------------------------------ accuracy: vs CANN baseline


@pytest.mark.parametrize("case", MODEL_SHAPES, ids=lambda c: c[0])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_accuracy_vs_cann(case, output_dtype):
    """Triton kernel output matches CANN baseline for profiled model shapes."""
    name, M, K, N, nz = case
    x, sa, w, w_nd, sb = _build_mxfp8_inputs(M, K, N, nz=nz)

    base = _cann_baseline(x, sa, w, sb, output_dtype=output_dtype)
    got = flag_gems.quant_batch_matmul_mxfp8(
        x, w, sb, sa, output_dtype=output_dtype,
    )
    torch.npu.synchronize()

    assert got.shape == base.shape
    assert got.dtype == output_dtype

    # max_rel threshold: bf16 output with large K accumulation allows 2e-2,
    # but in practice we often get bit-exact (max_rel=0) for non-split-K cases.
    diff = (got.float() - base.float()).abs()
    denom = base.float().abs().max().clamp(min=1e-6)
    max_rel = (diff.max() / denom).item()
    assert max_rel <= 2e-2, (
        f"{name}: max_rel={max_rel:.3e} exceeds threshold 2e-2"
    )


@pytest.mark.parametrize("case", BOUNDARY_SHAPES, ids=lambda c: c[0])
def test_accuracy_boundary_shapes(case):
    """Boundary and edge-case shapes produce correct results."""
    name, M, K, N, nz = case
    x, sa, w, w_nd, sb = _build_mxfp8_inputs(M, K, N, nz=nz)

    # K=64 baseline doesn't work with CANN (ERR01001), use torch ref only
    has_cann = True
    try:
        base = _cann_baseline(x, sa, w, sb)
    except RuntimeError:
        has_cann = False

    got = flag_gems.quant_batch_matmul_mxfp8(x, w, sb, sa)
    torch.npu.synchronize()

    assert got.shape == (M, N)
    assert got.dtype == torch.bfloat16

    if has_cann:
        diff = (got.float() - base.float()).abs()
        denom = base.float().abs().max().clamp(min=1e-6)
        max_rel = (diff.max() / denom).item()
        assert max_rel <= 2e-2, (
            f"{name}: max_rel={max_rel:.3e} vs CANN exceeds 2e-2"
        )

    # Cross-validate with torch reference (ND only)
    if not nz:
        ref = _torch_reference(x, w_nd, sb, sa)
        diff_ref = (got.float() - ref.float()).abs()
        denom_ref = ref.float().abs().max().clamp(min=1e-6)
        max_rel_ref = (diff_ref.max() / denom_ref).item()
        assert max_rel_ref <= 2e-2, (
            f"{name}: max_rel={max_rel_ref:.3e} vs torch ref exceeds 2e-2"
        )


# ------------------------------------------------ accuracy: vs torch reference


@pytest.mark.parametrize(
    "M,K,N",
    [(1, 2048, 1024), (4, 1024, 512), (1, 2048, 16384)],
    ids=["decode", "small-batch", "wide"],
)
def test_accuracy_vs_torch_ref(M, K, N):
    """Triton output matches pure-torch fp32 reference (ND weights)."""
    x, sa, w, w_nd, sb = _build_mxfp8_inputs(M, K, N, nz=False)
    got = flag_gems.quant_batch_matmul_mxfp8(x, w, sb, sa)
    ref = _torch_reference(x, w_nd, sb, sa)
    torch.npu.synchronize()

    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().max().clamp(min=1e-6)
    max_rel = (diff.max() / denom).item()
    assert max_rel <= 2e-2


# ------------------------------------------------ bias support


def test_bias():
    """Output bias is correctly applied, validated against torch fp32 reference."""
    M, K, N = 1, 2048, 1024
    x, sa, w, w_nd, sb = _build_mxfp8_inputs(M, K, N, nz=False)
    bias = torch.randn(N, device=flag_gems.device, dtype=torch.float32) * 0.01

    got = flag_gems.quant_batch_matmul_mxfp8(x, w, sb, sa, bias=bias)
    ref = _torch_reference(x, w_nd, sb, sa, output_dtype=torch.float32)
    expected = (ref + bias.unsqueeze(0)).to(torch.bfloat16)
    torch.npu.synchronize()

    diff = (got.float() - expected.float()).abs()
    denom = expected.float().abs().max().clamp(min=1e-6)
    max_rel = (diff.max() / denom).item()
    assert max_rel <= 2e-2, f"bias test: max_rel={max_rel:.3e} exceeds 2e-2"

    # Also verify bias actually changes the output
    got_no_bias = flag_gems.quant_batch_matmul_mxfp8(x, w, sb, sa)
    torch.npu.synchronize()
    assert not torch.equal(got, got_no_bias), "bias should change the output"


# ------------------------------------------------ NZ auto-detection


def test_nz_auto_detection():
    """NZ vs ND is correctly auto-detected; both produce matching results."""
    M, K, N = 1, 2048, 1024
    x, sa, w_nz, w_nd, sb = _build_mxfp8_inputs(M, K, N, nz=True)

    got_nz = flag_gems.quant_batch_matmul_mxfp8(x, w_nz, sb, sa)
    got_nd = flag_gems.quant_batch_matmul_mxfp8(x, w_nd, sb, sa)
    torch.npu.synchronize()

    # NZ and ND should give the same result since dot_scaled is bit-exact
    diff = (got_nz.float() - got_nd.float()).abs()
    denom = got_nd.float().abs().max().clamp(min=1e-6)
    max_rel = (diff.max() / denom).item()
    assert max_rel <= 2e-2, (
        f"NZ vs ND mismatch: max_rel={max_rel:.3e}"
    )


# ------------------------------------------------ input validation


@pytest.mark.parametrize(
    "bad_input,match",
    [
        ("x_3d", "must be 2-D"),
        ("w_3d", "must be 2-D"),
        ("k_mismatch", "K mismatch"),
        ("wrong_dtype", "float8_e4m3fn"),
        ("k_not_32", "multiple of 32"),
        ("sb_wrong_numel", "weight_scale elements"),
        ("sa_wrong_numel", "pertoken_scale elements"),
    ],
)
def test_input_validation(bad_input, match):
    """Invalid inputs are rejected with descriptive error messages."""
    dev = flag_gems.device
    M, K, N = 1, 2048, 1024
    x = torch.zeros(M, K, device=dev, dtype=torch.float8_e4m3fn)
    w = torch.zeros(K, N, device=dev, dtype=torch.float8_e4m3fn)
    sb = torch.zeros((K // 32) * N, device=dev, dtype=torch.uint8)
    sa = torch.zeros(M * (K // 32), device=dev, dtype=torch.uint8)

    if bad_input == "x_3d":
        x = x.unsqueeze(0)
    elif bad_input == "w_3d":
        w = w.unsqueeze(0)
    elif bad_input == "k_mismatch":
        w = torch.zeros(K + 64, N, device=dev, dtype=torch.float8_e4m3fn)
    elif bad_input == "wrong_dtype":
        x = torch.zeros(M, K, device=dev, dtype=torch.bfloat16)
    elif bad_input == "k_not_32":
        x = torch.zeros(M, 30, device=dev, dtype=torch.float8_e4m3fn)
        w = torch.zeros(30, N, device=dev, dtype=torch.float8_e4m3fn)
    elif bad_input == "sb_wrong_numel":
        sb = torch.zeros(10, device=dev, dtype=torch.uint8)
    elif bad_input == "sa_wrong_numel":
        sa = torch.zeros(1, device=dev, dtype=torch.uint8)

    with pytest.raises((ValueError, TypeError), match=match):
        flag_gems.quant_batch_matmul_mxfp8(x, w, sb, sa)


# ------------------------------------------------ module internals


@pytest.fixture
def qbmm_module():
    return importlib.import_module(flag_gems.quant_batch_matmul_mxfp8.__module__)


def test_pick_tiling_defaults(qbmm_module):
    """Default tiling picks reasonable values for a typical shape."""
    bn, sk, bk = qbmm_module._pick_tiling(
        M=1, K=2048, N=16384, cores=32,
        block_n=None, split_k=None, block_k=None,
    )
    assert 16384 % bn == 0, f"BLOCK_N={bn} doesn't divide N=16384"
    assert bk % 64 == 0, f"BLOCK_K={bk} not multiple of 64"
    assert sk >= 1


def test_pick_tiling_rejects_multi_mtile_splitk(qbmm_module):
    """split-K is rejected when M needs multiple tiles (compiler limitation)."""
    with pytest.raises(ValueError, match="m-tile"):
        qbmm_module._pick_tiling(
            M=32, K=2048, N=1024, cores=32,
            block_n=None, split_k=4, block_k=None,
        )


def test_is_nz_nd(qbmm_module):
    """ND tensor is correctly identified as non-NZ."""
    t = torch.zeros(64, 64, device=flag_gems.device, dtype=torch.float8_e4m3fn)
    assert not qbmm_module._is_nz(t)


def test_is_nz_fractal(qbmm_module):
    """FRACTAL_NZ tensor is correctly identified."""
    t = torch.zeros(64, 64, device=flag_gems.device, dtype=torch.float8_e4m3fn)
    t_nz = torch_npu.npu_format_cast(
        t, ACL_FORMAT_FRACTAL_NZ, customize_dtype=torch.float8_e4m3fn
    )
    assert qbmm_module._is_nz(t_nz)
