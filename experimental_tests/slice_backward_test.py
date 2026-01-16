# SLICE_BACKWARD operator test

import os
import sys

import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.slice_backward import (
    slice_backward as gems_slice_backward,
)
from flag_gems.experimental_ops.slice_backward import (
    slice_backward_out as gems_slice_backward_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
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


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (17, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_slice_backward_tensor(shape, dtype):
    input_sizes = list(shape)

    def test_cases_for_size(n):
        c = []
        c.append((0, n, 1))
        c.append((0, n, 2))
        if n > 1:
            c.append((1, n, 1))
        if n > 0:
            c.append((0, max(n - 1, 0), 1))
        if n > 3:
            c.append((1, n - 1, 2))
        # Filter duplicates and invalid
        uniq = []
        seen = set()
        for s, e, st in c:
            if st <= 0:
                continue
            if e <= s:
                continue
            key = (s, e, st)
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    for dim in range(len(input_sizes)):
        size_d = input_sizes[dim]
        for start, end, step in test_cases_for_size(size_d):
            length = (end - start + step - 1) // step
            if length <= 0:
                continue
            grad_shape = list(input_sizes)
            grad_shape[dim] = length
            grad_output = torch.randn(grad_shape, device=flag_gems.device, dtype=dtype)
            grad_output_ref = to_reference(grad_output)
            grad_output_act = grad_output.clone()

            ref_out = torch.ops.aten.slice_backward(
                grad_output_ref, input_sizes, dim, start, end, step
            )
            with flag_gems.use_gems():
                act_out = gems_slice_backward(
                    grad_output_act, input_sizes, dim, start, end, step
                )

            gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (17, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_slice_backward_out(shape, dtype):
    input_sizes = list(shape)

    def test_cases_for_size(n):
        c = []
        c.append((0, n, 1))
        c.append((0, n, 2))
        if n > 1:
            c.append((1, n, 1))
        if n > 0:
            c.append((0, max(n - 1, 0), 1))
        if n > 3:
            c.append((1, n - 1, 2))
        uniq = []
        seen = set()
        for s, e, st in c:
            if st <= 0:
                continue
            if e <= s:
                continue
            key = (s, e, st)
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    for dim in range(len(input_sizes)):
        size_d = input_sizes[dim]
        for start, end, step in test_cases_for_size(size_d):
            length = (end - start + step - 1) // step
            if length <= 0:
                continue
            grad_shape = list(input_sizes)
            grad_shape[dim] = length
            grad_output = torch.randn(grad_shape, device=flag_gems.device, dtype=dtype)
            grad_output_ref = to_reference(grad_output)
            grad_output_act = grad_output.clone()

            ref_out_buf = torch.empty(
                input_sizes, device=grad_output_ref.device, dtype=dtype
            )
            act_out_buf = torch.empty(input_sizes, device=flag_gems.device, dtype=dtype)

            ref_out = torch.ops.aten.slice_backward.out(
                grad_output_ref, input_sizes, dim, start, end, step, out=ref_out_buf
            )
            with flag_gems.use_gems():
                act_out = gems_slice_backward_out(
                    grad_output_act, input_sizes, dim, start, end, step, act_out_buf
                )

            gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (17, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_slice_backward_benchmark_tensor(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    input_sizes = list(shape)

    def test_cases_for_size(n):
        c = []
        c.append((0, n, 1))
        c.append((0, n, 2))
        if n > 1:
            c.append((1, n, 1))
        if n > 0:
            c.append((0, max(n - 1, 0), 1))
        if n > 3:
            c.append((1, n - 1, 2))
        # Filter duplicates and invalid
        uniq = []
        seen = set()
        for s, e, st in c:
            if st <= 0:
                continue
            if e <= s:
                continue
            key = (s, e, st)
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    for dim in range(len(input_sizes)):
        size_d = input_sizes[dim]
        for start, end, step in test_cases_for_size(size_d):
            length = (end - start + step - 1) // step
            if length <= 0:
                continue
            grad_shape = list(input_sizes)
            grad_shape[dim] = length
            grad_output_ref = torch.randn(
                grad_shape, device=flag_gems.device, dtype=dtype
            )
            grad_output_act = grad_output_ref.clone()

            # PyTorch reference implementation
            ms_torch, _, _ = triton.testing.do_bench(
                lambda: torch.ops.aten.slice_backward(
                    grad_output_ref, input_sizes, dim, start, end, step
                ),
                rep=100,
                quantiles=quantiles,
            )

            # Triton implementation
            with flag_gems.use_gems():
                ms_triton, _, _ = triton.testing.do_bench(
                    lambda: gems_slice_backward(
                        grad_output_act, input_sizes, dim, start, end, step
                    ),
                    rep=100,
                    quantiles=quantiles,
                )

            # Calculate speedup and return result
            speedup = ms_torch / ms_triton

            print(f"slice_backward {shape} {dtype}:")
            print(f"  FlagGems: {ms_triton:.3f}ms")
            print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.slice_backward
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512), (17, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_slice_backward_benchmark_out(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    input_sizes = list(shape)

    def test_cases_for_size(n):
        c = []
        c.append((0, n, 1))
        c.append((0, n, 2))
        if n > 1:
            c.append((1, n, 1))
        if n > 0:
            c.append((0, max(n - 1, 0), 1))
        if n > 3:
            c.append((1, n - 1, 2))
        uniq = []
        seen = set()
        for s, e, st in c:
            if st <= 0:
                continue
            if e <= s:
                continue
            key = (s, e, st)
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    for dim in range(len(input_sizes)):
        size_d = input_sizes[dim]
        for start, end, step in test_cases_for_size(size_d):
            length = (end - start + step - 1) // step
            if length <= 0:
                continue
            grad_shape = list(input_sizes)
            grad_shape[dim] = length
            grad_output_ref = torch.randn(
                grad_shape, device=flag_gems.device, dtype=dtype
            )
            grad_output_act = grad_output_ref.clone()

            ref_out_buf = torch.empty(input_sizes, device=flag_gems.device, dtype=dtype)
            act_out_buf = torch.empty(input_sizes, device=flag_gems.device, dtype=dtype)

            # PyTorch reference implementation
            ms_torch, _, _ = triton.testing.do_bench(
                lambda: torch.ops.aten.slice_backward.out(
                    grad_output_ref, input_sizes, dim, start, end, step, out=ref_out_buf
                ),
                rep=100,
                quantiles=quantiles,
            )

            # Triton implementation
            with flag_gems.use_gems():
                ms_triton, _, _ = triton.testing.do_bench(
                    lambda: gems_slice_backward_out(
                        grad_output_act, input_sizes, dim, start, end, step, act_out_buf
                    ),
                    rep=100,
                    quantiles=quantiles,
                )

            # Calculate speedup and return result
            speedup = ms_torch / ms_triton

            print(f"slice_backward {shape} {dtype}:")
            print(f"  FlagGems: {ms_triton:.3f}ms")
            print(f"  Speedup: {speedup:.2f}x")
