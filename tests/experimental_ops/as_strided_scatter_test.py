# AS_STRIDED_SCATTER operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.as_strided_scatter import (
    as_strided_scatter as gems_as_strided_scatter,
)
from flag_gems.experimental_ops.as_strided_scatter import (
    as_strided_scatter_out as gems_as_strided_scatter_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.as_strided_scatter
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "spec",
    [
        {"size": (2, 3), "stride": (3, 1), "storage_offset": 0, "self_len": 6},
        {
            "size": (128, 256),
            "stride": (256, 1),
            "storage_offset": 0,
            "self_len": 128 * 256,
        },
        {"size": (4, 4), "stride": (10, 2), "storage_offset": 3, "self_len": 40},
        {"size": (32, 32), "stride": (70, 2), "storage_offset": 5, "self_len": 2240},
        {"size": (100,), "stride": (2,), "storage_offset": 1, "self_len": 200},
    ],
)
def test_as_strided_scatter_tensor(dtype, spec):
    size = list(spec["size"])
    stride = list(spec["stride"])
    storage_offset = spec["storage_offset"]
    self_len = spec["self_len"]

    self_tensor = torch.randn((self_len,), dtype=dtype, device=flag_gems.device)
    src = torch.randn(size, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self_tensor)
    ref_src = to_reference(src)
    ref_out = torch.ops.aten.as_strided_scatter(
        ref_self, ref_src, size, stride, storage_offset
    )

    act_self = self_tensor.clone()
    act_src = src.clone()
    with flag_gems.use_gems():
        act_out = gems_as_strided_scatter(
            act_self, act_src, size, stride, storage_offset
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided_scatter
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "spec",
    [
        {"size": (2, 3), "stride": (3, 1), "storage_offset": 0, "self_len": 6},
        {
            "size": (128, 256),
            "stride": (256, 1),
            "storage_offset": 0,
            "self_len": 128 * 256,
        },
        {"size": (4, 4), "stride": (10, 2), "storage_offset": 3, "self_len": 40},
        {"size": (32, 32), "stride": (70, 2), "storage_offset": 5, "self_len": 2240},
        {"size": (100,), "stride": (2,), "storage_offset": 1, "self_len": 200},
    ],
)
def test_as_strided_scatter_out(dtype, spec):
    size = list(spec["size"])
    stride = list(spec["stride"])
    storage_offset = spec["storage_offset"]
    self_len = spec["self_len"]

    self_tensor = torch.randn((self_len,), dtype=dtype, device=flag_gems.device)
    src = torch.randn(size, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self_tensor)
    ref_src = to_reference(src)
    ref_out_buf = torch.empty_like(ref_self)
    ref_out = torch.ops.aten.as_strided_scatter.out(
        ref_self, ref_src, size, stride, storage_offset, out=ref_out_buf
    )

    act_self = self_tensor.clone()
    act_src = src.clone()
    act_out_buf = torch.empty_like(act_self)
    with flag_gems.use_gems():
        act_out = gems_as_strided_scatter_out(
            act_self, act_src, size, stride, storage_offset, act_out_buf
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.as_strided_scatter
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "spec",
    [
        {"size": (2, 3), "stride": (3, 1), "storage_offset": 0, "self_len": 6},
        {
            "size": (128, 256),
            "stride": (256, 1),
            "storage_offset": 0,
            "self_len": 128 * 256,
        },
        {"size": (4, 4), "stride": (10, 2), "storage_offset": 3, "self_len": 40},
        {"size": (32, 32), "stride": (70, 2), "storage_offset": 5, "self_len": 2240},
        {"size": (100,), "stride": (2,), "storage_offset": 1, "self_len": 200},
    ],
)
def test_as_strided_scatter_benchmark_tensor(dtype, spec):
    quantiles = [0.5, 0.2, 0.8]

    size = list(spec["size"])
    stride = list(spec["stride"])
    storage_offset = spec["storage_offset"]
    self_len = spec["self_len"]

    self_tensor = torch.randn((self_len,), dtype=dtype, device=flag_gems.device)
    src = torch.randn(size, dtype=dtype, device=flag_gems.device)

    ref_self = self_tensor.clone()
    ref_src = src.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided_scatter(
            ref_self, ref_src, size, stride, storage_offset
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_self = self_tensor.clone()
    act_src = src.clone()

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided_scatter(
                act_self, act_src, size, stride, storage_offset
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided_scatter {spec['size']} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.as_strided_scatter
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "spec",
    [
        {"size": (2, 3), "stride": (3, 1), "storage_offset": 0, "self_len": 6},
        {
            "size": (128, 256),
            "stride": (256, 1),
            "storage_offset": 0,
            "self_len": 128 * 256,
        },
        {"size": (4, 4), "stride": (10, 2), "storage_offset": 3, "self_len": 40},
        {"size": (32, 32), "stride": (70, 2), "storage_offset": 5, "self_len": 2240},
        {"size": (100,), "stride": (2,), "storage_offset": 1, "self_len": 200},
    ],
)
def test_as_strided_scatter_benchmark_out(dtype, spec):
    quantiles = [0.5, 0.2, 0.8]

    size = list(spec["size"])
    stride = list(spec["stride"])
    storage_offset = spec["storage_offset"]
    self_len = spec["self_len"]

    self_tensor = torch.randn((self_len,), dtype=dtype, device=flag_gems.device)
    src = torch.randn(size, dtype=dtype, device=flag_gems.device)

    ref_self = self_tensor.clone()
    ref_src = src.clone()
    ref_out_buf = torch.empty_like(ref_self)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.as_strided_scatter.out(
            ref_self, ref_src, size, stride, storage_offset, out=ref_out_buf
        ),
        rep=100,
        quantiles=quantiles,
    )

    act_self = self_tensor.clone()
    act_src = src.clone()
    act_out_buf = torch.empty_like(act_self)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_as_strided_scatter_out(
                act_self, act_src, size, stride, storage_offset, act_out_buf
            ),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"as_strided_scatter {spec['size']} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
