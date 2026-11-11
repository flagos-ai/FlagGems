"""Numerical and layout regressions for the Hygon small-kernel paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)

@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("shape", [(), (6,), (33, 78), (542, 30), (101, 3, 33)])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "vector"])
def test_hygon_flat_add(dtype, shape, kind):
    a = torch.randint(-20, 20, shape).to(dtype)
    b = (
        3
        if kind == "scalar"
        else torch.randint(-20, 20, shape if kind == "tensor" else shape[-1:]).to(dtype)
    )
    alpha = -2
    reference = torch.add(a, b, alpha=alpha)
    result = flag_gems.add(
        a.to(flag_gems.device),
        b.to(flag_gems.device) if isinstance(b, torch.Tensor) else b,
        alpha=alpha,
    ).cpu()
    torch.testing.assert_close(result, reference)

@pytest.mark.add
@pytest.mark.parametrize("kind", ["transpose", "broadcast", "slice"])
def test_hygon_strided_add(kind):
    base = torch.randn(2, 3, 37, 41, dtype=torch.float64)
    a = base[:, :, :33, :33]
    if kind == "transpose":
        b = a.transpose(-1, -2)
    elif kind == "broadcast":
        b = a[:, :, :1, :]
    else:
        b = base[:, :, 1:34, 1:34]
    gpu_base = base.to(flag_gems.device)
    gpu_a = torch.as_strided(gpu_base, a.shape, a.stride(), a.storage_offset())
    gpu_b = torch.as_strided(gpu_base, b.shape, b.stride(), b.storage_offset())
    result = flag_gems.add(gpu_a, gpu_b, alpha=-3)
    torch.testing.assert_close(result.cpu(), torch.add(a, b, alpha=-3))

@pytest.mark.add
@pytest.mark.parametrize("size", [24575, 24576, 32768, 32769])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "broadcast"])
def test_hygon_medium_flat_add(size, kind):
    generator = torch.Generator().manual_seed(6801)
    a = torch.randn(size, generator=generator) * 0.25
    b = (
        0
        if kind == "scalar"
        else torch.randn(1 if kind == "broadcast" else size, generator=generator) * 0.25
    )
    bg = b.to(flag_gems.device) if isinstance(b, torch.Tensor) else b
    actual = flag_gems.add(a.to(flag_gems.device), bg, alpha=-2)
    torch.testing.assert_close(actual.cpu(), torch.add(a, b, alpha=-2))

@pytest.mark.addmm
@pytest.mark.parametrize("shape", [(33, 112, 48), (63, 128, 31)])
@pytest.mark.parametrize("kind", ["plain", "strided", "scaled"])
def test_hygon_addmm_small_fp32_dispatch(shape, kind):
    m, k, n = shape
    g = torch.Generator().manual_seed(3191)
    a = (
        torch.randn(k, m, generator=g).t()
        if kind == "strided"
        else torch.randn(m, k, generator=g)
    ) * 0.25
    b = torch.randn(n, k, generator=g).t() * 0.25
    bias = torch.randn(n, generator=g) * 0.25
    alpha, beta = (0.5, -2) if kind == "scaled" else (1, 1)
    expected = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
    ag, bg, ig = (t.to(flag_gems.device) for t in (a, b, bias))
    if kind == "strided":
        out = torch.empty((m, n * 2), device=flag_gems.device)[:, ::2]
        result = flag_gems.addmm_out(ig, ag, bg, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(ig, ag, bg, alpha=alpha, beta=beta)
    torch.testing.assert_close(result.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize("seed", [2901, 2902])
@pytest.mark.parametrize("kind", ["dual", "strided_out", "scaled"])
def test_hygon_addmm_large_bf16_dispatch(seed, kind):
    # An incomplete M tile checks discarded-row loads in the dual accumulator.
    m, k, n = 8195, 2048, 2048
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    alpha, beta = (0.5, -1) if kind == "scaled" else (1, 1)
    expected = torch.addmm(
        bias.float(), a.float(), b.float(), alpha=alpha, beta=beta
    ).to(torch.bfloat16)
    ag, bg, ig = (t.to(flag_gems.device) for t in (a, b, bias))
    if kind == "strided_out":
        out = torch.empty((m, n * 2), dtype=torch.bfloat16, device=flag_gems.device)[
            :, ::2
        ]
        result = flag_gems.addmm_out(ig, ag, bg, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(ig, ag, bg, alpha=alpha, beta=beta)
    torch.testing.assert_close(result.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize("n", [4096, 6144])
@pytest.mark.parametrize("seed", [3901, 3902])
def test_hygon_addmm_bf16_quarter_stores(n, seed):
    m, k = 8195, 2048
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    )
    torch.testing.assert_close(result.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(8195, 2112, 2304), (8195, 4096, 4096), (8447, 5120, 2048)]
)
@pytest.mark.parametrize("seed", [4901, 4902])
def test_hygon_addmm_bf16_long_k_half_stores(shape, seed):
    m, k, n = shape
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    )
    torch.testing.assert_close(result.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize("seed", [158000, 158001, 158002])
def test_hygon_addmm_fp32_accumulation(seed):
    g = torch.Generator().manual_seed(seed)
    bias = torch.randn(1152, generator=g) * 0.25
    a = torch.randn(400, 1280, generator=g) * 0.25
    b = (torch.randn(1152, 1280, generator=g) * 0.25).t()
    expected = torch.addmm(bias, a, b)
    result = flag_gems.addmm(
        bias.to(flag_gems.device), a.to(flag_gems.device), b.to(flag_gems.device)
    ).cpu()
    torch.testing.assert_close(result, expected)

@pytest.mark.addmm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta", [0, 1])
def test_hygon_addmm_low_precision_tails(dtype, beta):
    torch.manual_seed(905)
    a = (torch.randn(1031, 64) * 0.25).to(dtype)
    b = (torch.randn(257, 64) * 0.25).to(dtype).t()
    bias = (
        torch.full((257,), float("nan"), dtype=dtype)
        if beta == 0
        else torch.randn(257).to(dtype)
    )
    expected = torch.addmm(bias.float(), a.float(), b.float(), beta=beta).to(dtype)
    result = flag_gems.addmm(
        bias.to(flag_gems.device),
        a.to(flag_gems.device),
        b.to(flag_gems.device),
        beta=beta,
    )
    torch.testing.assert_close(result.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(128, 256, 256), (257, 289, 317), (303, 320, 320), (511, 511, 512)]
)
@pytest.mark.parametrize("scale", [0.25, 1.0])
@pytest.mark.parametrize("strided_output", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_hygon_addmm_medium_precise(shape, scale, strided_output, transpose_a):
    m, k, n = shape
    g = torch.Generator().manual_seed(1143)
    ac = (
        torch.randn(k, m, generator=g).t()
        if transpose_a
        else torch.randn(m, k, generator=g)
    ) * scale
    bc = torch.randn(n, k, generator=g).t() * scale
    bias = torch.randn(1, n, generator=g) * scale
    a = ac.to(flag_gems.device)
    b = bc.to(flag_gems.device)
    gpu_bias = bias.to(flag_gems.device)
    # Use double precision on CPU so the reference does not introduce FP32 GEMM error.
    expected = torch.addmm(
        bias.double(), ac.double(), bc.double(), alpha=-0.5, beta=1.25
    ).float()
    if strided_output:
        out = torch.empty(n, m, device=flag_gems.device).t()
        result = flag_gems.addmm_out(gpu_bias, a, b, alpha=-0.5, beta=1.25, out=out)
        assert result.data_ptr() == out.data_ptr()
    else:
        result = flag_gems.addmm(gpu_bias, a, b, alpha=-0.5, beta=1.25)
    torch.testing.assert_close(result.cpu(), expected)
    nan_bias = torch.full((n,), float("nan"))
    result = flag_gems.addmm(nan_bias.to(flag_gems.device), a, b, beta=0)
    expected = torch.addmm(nan_bias.double(), ac.double(), bc.double(), beta=0).float()
    torch.testing.assert_close(result.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize(
    "unsafe", ["none", "register_use", "missing_final_wait", "global_store"]
)
def test_hygon_addmm_schedule_dependencies(unsafe):
    from flag_gems.runtime.backend._hygon.ops.addmm_schedule import _defer_prefetch_wait

    assembly = """loop: ; Inner Loop Header
    buffer_load_dwordx4 v[140:143], v172, s[0:3], 0 offen
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    ds_read_b128 v[144:147], v134
    buffer_load_dwordx4 v[168:171], v176, s[4:7], 0 offen
    buffer_load_dwordx4 v[172:175], v172, s[0:3], 0 offen offset:64
    buffer_load_dwordx4 v[176:179], v176, s[4:7], 0 offen offset:64
    v_mmac_f32_16x16x16_bf16 v[61:64], v[144:145], v[148:149], v[61:64]
    s_waitcnt vmcnt(0) lgkmcnt(0)
    s_barrier
    ds_write_b128 v137, v[140:143]
    ds_write_b128 v137, v[168:171] offset:16384
    ds_write_b128 v137, v[172:175] offset:32768
    ds_write_b128 v137, v[176:179] offset:49152
    s_cbranch_scc1 loop
"""
    if unsafe == "register_use":
        assembly = assembly.replace(
            "    ds_read_b128", "    v_mov_b32 v145, v140\n    ds_read_b128"
        )
    elif unsafe == "missing_final_wait":
        at = assembly.rindex("s_waitcnt vmcnt(0)")
        assembly = assembly[:at] + assembly[at:].replace("vmcnt(0)", "vmcnt(1)", 1)
    elif unsafe == "global_store":
        assembly = assembly.replace(
            "    ds_read_b128",
            "    buffer_store_dword v145, v172, s[0:3], 0 offen\n    ds_read_b128",
        )
    result = _defer_prefetch_wait(assembly)
    if unsafe != "none":
        assert result == assembly
    else:
        assert result == assembly.replace(
            "s_waitcnt vmcnt(0) lgkmcnt(0)", "s_waitcnt lgkmcnt(0)", 1
        )
        assert result.count("s_barrier") == 2

@pytest.mark.addmm
@pytest.mark.parametrize("unaligned", ["bias", "out"])
def test_hygon_addmm_long_k_unaligned(unaligned):
    m, k, n = 8195, 2112, 2304
    g = torch.Generator().manual_seed(6301)
    a = (torch.randn(m, k, generator=g) * 0.25).to(torch.bfloat16)
    b = (torch.randn(n, k, generator=g) * 0.25).to(torch.bfloat16).t()
    bias = (torch.randn(n, generator=g) * 0.25).to(torch.bfloat16)
    expected = torch.addmm(bias.float(), a.float(), b.float()).to(torch.bfloat16)
    ag, bg, ig = (t.to(flag_gems.device) for t in (a, b, bias))
    if unaligned == "bias":
        ig = torch.empty(n + 1, dtype=torch.bfloat16, device=flag_gems.device)[1:]
        ig.copy_(bias)
    extra = 1 if unaligned == "out" else 0
    out = torch.empty(m * n + extra, dtype=torch.bfloat16, device=flag_gems.device)[
        extra:
    ].view(m, n)
    result = flag_gems.addmm_out(ig, ag, bg, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result.cpu(), expected)

"""Regression coverage for expanded Hygon layouts and precise reductions."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)

@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(1, 1024, 37), (1, 8193, 1), (17, 4097, 1), (333, 83, 63)]
)
@pytest.mark.parametrize("beta", [0, 1])
def test_expanded_addmm_precision_and_output_stride(shape, beta):
    m, k, n = shape
    g = torch.Generator().manual_seed(702)
    a = torch.randn(m, k, generator=g) * 0.25
    b = torch.randn(k, n, generator=g) * 0.25
    bias = torch.full((n,), float("nan")) if beta == 0 else torch.randn(n, generator=g)
    expected = torch.addmm(
        bias.double(), a.double(), b.double(), alpha=2, beta=beta
    ).float()
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    out = torch.empty((m, 2 * n), device=flag_gems.device)[:, ::2]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd, alpha=2, beta=beta)
        returned = flag_gems.addmm_out(cd, ad, bd, alpha=2, beta=beta, out=out)
    assert returned is out
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(out.cpu(), expected)

@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "layout",
    ["transpose_broadcast", "transpose_dense", "offset_rows", "broadcast_plane"],
)
def test_expanded_add_layouts(dtype, layout):
    g = torch.Generator().manual_seed(703)
    if layout.startswith("transpose"):
        backing = torch.randn(2, 129, 1025, generator=g, dtype=dtype)
        a = backing.transpose(1, 2)
        b = torch.randn(
            (2, 1, 129) if layout.endswith("broadcast") else (2, 1025, 129),
            generator=g,
            dtype=dtype,
        )
        ad = backing.to(flag_gems.device).transpose(1, 2)
        bd = b.to(flag_gems.device)
    elif layout == "offset_rows":
        backing = torch.randn(257, 4, 1159, generator=g, dtype=dtype)
        a, b = backing[:, 1:2, :], backing[:, 2:3, :]
        storage = backing.to(flag_gems.device)
        ad, bd = storage[:, 1:2, :], storage[:, 2:3, :]
    else:
        a = torch.randn(7, 197, 199, generator=g, dtype=dtype)
        b = torch.randn(7, 1, 1, generator=g, dtype=dtype)
        ad, bd = a.to(flag_gems.device), b.to(flag_gems.device)
    with torch.no_grad():
        actual = flag_gems.add(ad, bd, alpha=2).cpu()
    torch.testing.assert_close(actual, torch.add(a, b, alpha=2))

@pytest.mark.add
def test_expanded_add_int64_scalar():
    a = torch.tensor([-(2**40), 2**40], dtype=torch.int64)
    actual = flag_gems.add(a.to(flag_gems.device), 3, alpha=2).cpu()
    assert torch.equal(actual, a + 6)

@pytest.mark.addmm
@pytest.mark.parametrize("beta", [0, 1])
def test_expanded_addmm_aligned_k(beta):
    generator = torch.Generator().manual_seed(20261109)
    m, n, k = 4096, 512, 1025
    a = (torch.randn(m * k + 7, generator=generator) * 0.125)[3 : 3 + m * k].view(m, k)
    b = (
        (torch.randn(n * k + 9, generator=generator) * 0.125)[5 : 5 + n * k]
        .view(n, k)
        .t()
    )
    bias = torch.randn(n, generator=generator) * 0.125
    if beta == 0:
        bias.fill_(float("nan"))
    expected = torch.addmm(bias, a, b, alpha=0.5, beta=beta)
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    out = torch.empty((m, 2 * n), device=flag_gems.device)[:, ::2]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd, alpha=0.5, beta=beta)
        returned = flag_gems.addmm_out(cd, ad, bd, alpha=0.5, beta=beta, out=out)
    assert returned is out
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(out.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize(
    "kind,alpha,beta",
    [("vector", 1, 1), ("scalar", 0.5, -0.75), ("row", -1, 1), ("nan", 2, 0)],
)
def test_expanded_addmm_transposed_bias_and_out(kind, alpha, beta):
    m, k, n = 4097, 1025, 513
    g = torch.Generator().manual_seed(745)
    a = torch.randn(m, k, generator=g) * 0.25
    b = (torch.randn(n, k, generator=g) * 0.25).t()
    if kind == "scalar":
        bias = torch.tensor(0.25)
    elif kind == "row":
        bias = torch.randn(m, 1, generator=g)
    elif kind == "nan":
        bias = torch.full((n,), float("nan"))
    else:
        bias = torch.randn(n, generator=g)
    expected = torch.addmm(bias, a, b, alpha=alpha, beta=beta)
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    out = torch.empty((m, 2 * n + 3), device=flag_gems.device)[:, 1 : 2 * n + 1 : 2]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd, alpha=alpha, beta=beta)
        returned = flag_gems.addmm_out(cd, ad, bd, alpha=alpha, beta=beta, out=out)
    assert returned is out
    torch.testing.assert_close(actual.cpu(), expected)
    torch.testing.assert_close(out.cpu(), expected)

@pytest.mark.addmm
@pytest.mark.parametrize(
    "shape", [(512, 1024, 512), (800, 1024, 1024), (1024, 1024, 2048)]
)
@pytest.mark.parametrize("seed", [915, 916, 917])
def test_expanded_addmm_short_rowmajor_precision(shape, seed):
    m, k, n = shape
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g) * 0.25
    b = torch.randn(k, n, generator=g) * 0.25
    bias = torch.randn(n, generator=g) * 0.25
    expected = torch.addmm(bias, a, b)
    ad, bd, cd = [t.to(flag_gems.device) for t in (a, b, bias)]
    with torch.no_grad():
        actual = flag_gems.addmm(cd, ad, bd)
    torch.testing.assert_close(actual.cpu(), expected)
