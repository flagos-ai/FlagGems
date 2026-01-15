# STACK operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.stack import stack as gems_stack  # noqa: E402
from flag_gems.experimental_ops.stack import stack_out as gems_stack_out  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp):
    """Convert tensor to reference device (CPU if TO_CPU is True)."""
    if TO_CPU:
        return inp.to("cpu")
    return inp.clone()


@pytest.mark.stack
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (64, 32, 16), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tensors", [2, 3, 5])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack_tensor(shape, dtype, num_tensors, dim):
    tensors = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device)
        for _ in range(num_tensors)
    ]
    ref_tensors = [to_reference(t) for t in tensors]
    ref_out = torch.ops.aten.stack(ref_tensors, dim)
    with flag_gems.use_gems():
        act_out = gems_stack(tensors, dim)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.stack
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (64, 32, 16), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tensors", [2, 4])
@pytest.mark.parametrize("dim", [0, 2, -1])
def test_stack_out(shape, dtype, num_tensors, dim):
    tensors = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device)
        for _ in range(num_tensors)
    ]
    ref_tensors = [to_reference(t) for t in tensors]

    n_dims = len(shape)
    eff_dim = dim if dim >= 0 else dim + (n_dims + 1)
    out_shape = list(shape)
    out_shape.insert(eff_dim, num_tensors)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=ref_tensors[0].device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.stack.out(ref_tensors, dim, out=ref_out_buf)
    with flag_gems.use_gems():
        act_out = gems_stack_out(tensors, dim, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.stack
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (64, 32, 16), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tensors", [2, 3, 5])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack_benchmark_tensor(shape, dtype, num_tensors, dim):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401, F401

    quantiles = [0.5, 0.2, 0.8]

    tensors = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device)
        for _ in range(num_tensors)
    ]
    ref_tensors = [t.clone() for t in tensors]
    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.stack(ref_tensors, dim), rep=100, quantiles=quantiles
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_stack(tensors, dim), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"stack {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.stack
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (64, 32, 16), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tensors", [2, 4])
@pytest.mark.parametrize("dim", [0, 2, -1])
def test_stack_benchmark_out(shape, dtype, num_tensors, dim):
    quantiles = [0.5, 0.2, 0.8]

    tensors = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device)
        for _ in range(num_tensors)
    ]
    ref_tensors = [t.clone() for t in tensors]

    n_dims = len(shape)
    eff_dim = dim if dim >= 0 else dim + (n_dims + 1)
    out_shape = list(shape)
    out_shape.insert(eff_dim, num_tensors)

    ref_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    act_out_buf = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.stack.out(ref_tensors, dim, out=ref_out_buf),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_stack_out(tensors, dim, act_out_buf),
            rep=100,
            quantiles=quantiles,
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"stack {shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
