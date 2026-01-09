# SQUEEZE_COPY operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.squeeze_copy import (  # noqa: E402
    squeeze_copy as gems_squeeze_copy,
)
from flag_gems.experimental_ops.squeeze_copy import (  # noqa: E402
    squeeze_copy_out as gems_squeeze_copy_out,
)


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


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape", [(2, 3), (2, 1, 3, 1), (128, 256), (128, 1, 256), (512, 512), (512, 1, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_tensor(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.squeeze_copy(ref_x)
    with flag_gems.use_gems():
        act_out = gems_squeeze_copy(x)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dim",
    [
        ((2, 1, 3), 1),
        ((4, 5), 0),
        ((8, 1, 1, 2), -2),
        ((128, 1, 256), 1),
        ((32, 32), -1),
        ((512, 1, 64), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dim(shape_dim, dtype):
    shape, dim = shape_dim
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.squeeze_copy.dim(ref_x, dim)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.squeeze_copy.dim(x, dim)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dims",
    [
        ((2, 1, 3, 1), [1, 3]),
        ((1, 4, 1), [0, 2]),
        ((128, 1, 256, 1), [1]),
        ((64, 1, 1, 32), [1, -2]),
        ((16, 8, 4), [0]),
        ((512, 1, 64), [1]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dims(shape_dims, dtype):
    shape, dims = shape_dims
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten.squeeze_copy.dims(ref_x, dims)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.squeeze_copy.dims(x, dims)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape", [(2, 3), (2, 1, 3, 1), (128, 256), (128, 1, 256), (512, 512), (512, 1, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    tmp = torch.ops.aten.squeeze_copy(ref_x)
    ref_out_buf = torch.empty_like(tmp)
    ref_out = torch.ops.aten.squeeze_copy.out(ref_x, out=ref_out_buf)
    act_out_buf = torch.empty_like(tmp)
    with flag_gems.use_gems():
        act_out = gems_squeeze_copy_out(x, act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dim",
    [
        ((2, 1, 3), 1),
        ((4, 5), 0),
        ((8, 1, 1, 2), -2),
        ((128, 1, 256), 1),
        ((32, 32), -1),
        ((512, 1, 64), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dim_out(shape_dim, dtype):
    shape, dim = shape_dim
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    tmp = torch.ops.aten.squeeze_copy.dim(ref_x, dim)
    ref_out_buf = torch.empty_like(tmp)
    ref_out = torch.ops.aten.squeeze_copy.dim_out(ref_x, dim, out=ref_out_buf)
    act_out_buf = torch.empty_like(tmp)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.squeeze_copy.dim_out(x, dim, out=act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dims",
    [
        ((2, 1, 3, 1), [1, 3]),
        ((1, 4, 1), [0, 2]),
        ((128, 1, 256, 1), [1]),
        ((64, 1, 1, 32), [1, -2]),
        ((16, 8, 4), [0]),
        ((512, 1, 64), [1]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dims_out(shape_dims, dtype):
    shape, dims = shape_dims
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    tmp = torch.ops.aten.squeeze_copy.dims(ref_x, dims)
    ref_out_buf = torch.empty_like(tmp)
    ref_out = torch.ops.aten.squeeze_copy.dims_out(ref_x, dims, out=ref_out_buf)
    act_out_buf = torch.empty_like(tmp)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.squeeze_copy.dims_out(x, dims, out=act_out_buf)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape", [(2, 3), (2, 1, 3, 1), (128, 256), (128, 1, 256), (512, 512), (512, 1, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_benchmark_tensor(shape, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.squeeze_copy(ref_x), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_squeeze_copy(x), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"squeeze_copy {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dim",
    [
        ((2, 1, 3), 1),
        ((4, 5), 0),
        ((8, 1, 1, 2), -2),
        ((128, 1, 256), 1),
        ((32, 32), -1),
        ((512, 1, 64), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dim_performance(shape_dim, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dim = shape_dim
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.squeeze_copy.dim(ref_x, dim),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.squeeze_copy.dim(x, dim),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"squeeze_copy {shape_dim} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dims",
    [
        ((2, 1, 3, 1), [1, 3]),
        ((1, 4, 1), [0, 2]),
        ((128, 1, 256, 1), [1]),
        ((64, 1, 1, 32), [1, -2]),
        ((16, 8, 4), [0]),
        ((512, 1, 64), [1]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dims_performance(shape_dims, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dims = shape_dims
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.squeeze_copy.dims(ref_x, dims),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.squeeze_copy.dims(x, dims),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"squeeze_copy {shape_dims} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape", [(2, 3), (2, 1, 3, 1), (128, 256), (128, 1, 256), (512, 512), (512, 1, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_benchmark_out(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    tmp = torch.ops.aten.squeeze_copy(ref_x)
    ref_out_buf = torch.empty_like(tmp)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.squeeze_copy.out(ref_x, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )
    act_out_buf = torch.empty_like(tmp)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_squeeze_copy_out(x, act_out_buf), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"squeeze_copy {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dim",
    [
        ((2, 1, 3), 1),
        ((4, 5), 0),
        ((8, 1, 1, 2), -2),
        ((128, 1, 256), 1),
        ((32, 32), -1),
        ((512, 1, 64), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dim_benchmark_out(shape_dim, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dim = shape_dim
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    tmp = torch.ops.aten.squeeze_copy.dim(ref_x, dim)
    ref_out_buf = torch.empty_like(tmp)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.squeeze_copy.dim_out(ref_x, dim, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )
    act_out_buf = torch.empty_like(tmp)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.squeeze_copy.dim_out(x, dim, out=act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"squeeze_copy {shape_dim} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.squeeze_copy
@pytest.mark.parametrize(
    "shape_dims",
    [
        ((2, 1, 3, 1), [1, 3]),
        ((1, 4, 1), [0, 2]),
        ((128, 1, 256, 1), [1]),
        ((64, 1, 1, 32), [1, -2]),
        ((16, 8, 4), [0]),
        ((512, 1, 64), [1]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_squeeze_copy_dims_benchmark_out(shape_dims, dtype):
    quantiles = [0.5, 0.2, 0.8]

    shape, dims = shape_dims
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = x.clone()
    tmp = torch.ops.aten.squeeze_copy.dims(ref_x, dims)
    ref_out_buf = torch.empty_like(tmp)
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.squeeze_copy.dims_out(ref_x, dims, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )
    act_out_buf = torch.empty_like(tmp)

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.squeeze_copy.dims_out(x, dims, out=act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"squeeze_copy {shape_dims} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
