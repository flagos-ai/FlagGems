"""Numerical and layout regressions for the Hygon add implementation."""

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


@pytest.mark.add
@pytest.mark.parametrize("kind", ["medium_scalar", "large_broadcast", "scalar_tensor"])
def test_official_master_add_fallback_guards(monkeypatch, kind):
    import importlib

    impl = importlib.import_module("flag_gems.runtime.backend._hygon.ops.add")
    generic = impl._generic_add
    observed = []

    def checked_generic(a, b, *, alpha=1):
        observed.append(True)
        return generic(a, b, alpha=alpha)

    monkeypatch.setattr(impl, "_generic_add", checked_generic)
    shape = {
        "medium_scalar": (3, 101, 320),
        "large_broadcast": (1, 40000, 256),
        "scalar_tensor": (1025, 1024),
    }[kind]
    generator = torch.Generator().manual_seed(20261112)
    a = torch.randint(-20, 20, shape, generator=generator).float()
    if kind == "medium_scalar":
        b = 1.0
    elif kind == "large_broadcast":
        b = torch.randint(-20, 20, (1, 1, shape[-1]), generator=generator).float()
    else:
        b = torch.tensor(1.0)
    expected = torch.add(a, b, alpha=-2)
    actual = impl.add(
        a.to(flag_gems.device),
        b.to(flag_gems.device) if isinstance(b, torch.Tensor) else b,
        alpha=-2,
    )
    assert observed == [True], "regressed domains must retain the generic kernel"
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
