"""Numerical and layout regressions for the Hygon small-kernel paths."""

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon kernels"
)


@pytest.mark.clone
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
@pytest.mark.parametrize("kind", ["transpose", "slice", "broadcast", "empty"])
def test_hygon_clone_layout(memory_format, kind):
    base = torch.randn((7, 13, 4), dtype=torch.float64)
    if kind == "transpose":
        a = base.transpose(1, 2)
    elif kind == "slice":
        a = base[:, ::2, 1:]
    elif kind == "broadcast":
        a = base[:1].expand(5, 13, 4)
    else:
        a = base[:0]
    expected = torch.clone(a, memory_format=memory_format)
    # Rebuild the original view on device, including its holes and zero strides.
    gpu_base = base.to(flag_gems.device)
    gpu = torch.as_strided(gpu_base, a.shape, a.stride(), a.storage_offset())
    result = flag_gems.clone(gpu, memory_format=memory_format)
    assert result.shape == expected.shape
    assert result.stride() == expected.stride()
    assert torch.equal(result.cpu(), expected)


@pytest.mark.clone
@pytest.mark.parametrize(
    "shape, channels_format",
    [
        ((2, 3, 4, 5), torch.channels_last),
        ((2, 1, 4, 5), torch.channels_last),
        ((0, 3, 4, 5), torch.channels_last),
        ((2, 3, 2, 4, 5), torch.channels_last_3d),
        ((2, 1, 2, 4, 5), torch.channels_last_3d),
    ],
)
@pytest.mark.parametrize("target", ["default", "preserve", "contiguous", "channels"])
def test_hygon_clone_memory_format(shape, channels_format, target):
    formats = {
        "default": None,
        "preserve": torch.preserve_format,
        "contiguous": torch.contiguous_format,
        "channels": channels_format,
    }
    memory_format = formats[target]
    kwargs = {} if memory_format is None else {"memory_format": memory_format}
    cpu = torch.randn(shape, dtype=torch.float32).to(memory_format=channels_format)
    inp = torch.empty_strided(
        cpu.shape, cpu.stride(), dtype=cpu.dtype, device=flag_gems.device
    )
    inp.copy_(cpu)
    assert inp.stride() == cpu.stride()

    expected = torch.clone(cpu, **kwargs)
    actual = flag_gems.clone(inp, **kwargs)
    assert actual.stride() == expected.stride()
    assert actual.is_contiguous() == expected.is_contiguous()
    assert actual.is_contiguous(
        memory_format=channels_format
    ) == expected.is_contiguous(memory_format=channels_format)
    assert not torch._C._is_alias_of(actual, inp)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(inp.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.clone
def test_hygon_clone_meta_preserves_dense_strides():
    inp = torch.empty_strided((2, 1, 3), (3, 23, 1), device="meta")
    expected = torch.clone(inp)
    actual = flag_gems.clone(inp)
    assert actual.is_meta
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()


@pytest.mark.clone
@pytest.mark.parametrize("shape", [(1024, 1025), (17, 32, 64, 33)])
def test_hygon_clone_large_fp64_layout_conversion(shape):
    base = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    inp = base.transpose(-2, -1)
    expected = inp.cpu().clone(memory_format=torch.contiguous_format)
    actual = flag_gems.clone(inp, memory_format=torch.contiguous_format)
    assert actual.is_contiguous()
    assert not torch._C._is_alias_of(actual, inp)
    assert torch.equal(actual.cpu(), expected)
