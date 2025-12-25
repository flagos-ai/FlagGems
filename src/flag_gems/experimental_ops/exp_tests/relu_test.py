# ReLU operator test
# Please provide the test code here

# PyTorch baseline
import pytest
import torch
import triton

import flag_gems
from flag_gems.experimental_ops.relu import relu as gems_relu


def relu(input: torch.Tensor) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    dtype = input.dtype

    if input.is_complex():
        raise TypeError("relu does not support complex tensors.")

    if input.is_floating_point():
        # Use float32 for computation when input is lower precision
        if dtype in (torch.float16, torch.bfloat16):
            return torch.relu(input.to(torch.float32)).to(dtype)
        else:
            return torch.relu(input)

    if dtype == torch.bool:
        # For boolean tensors, ReLU is effectively identity
        return input.clone()

    # For integer tensors, use clamp_min to emulate ReLU
    return torch.clamp_min(input, 0)


@pytest.mark.relu
@pytest.mark.parametrize("shape", [(512,), (64, 128), (8, 32, 64), (2, 4, 8, 16), (1,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_relu_accuracy(shape, dtype):
    # initialize the input data based on the parameters
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # cast to reference dtype if necessary
    ref_input = input.cpu()

    ref_out = relu(ref_input)
    res_out = gems_relu(input)

    torch.testing.assert_close(res_out.cpu(), ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.relu
@pytest.mark.parametrize("shape", [(512,), (64, 128), (8, 32, 64), (2, 4, 8, 16), (1,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_relu_performance(shape, dtype):
    quantiles = [0.5, 0.2, 0.8]

    # initialize the input data based on the parameters
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # cast to reference dtype if necessary
    ref_input = input.clone()

    ms_torch, _, _ = triton.testing.do_bench(
        lambda: relu(ref_input), rep=100, quantiles=quantiles
    )
    ms_triton, _, _ = triton.testing.do_bench(
        lambda: gems_relu(input), rep=100, quantiles=quantiles
    )

    speedup = ms_torch / ms_triton

    print(f"relu {shape} {dtype}:")
    print(f"  PyTorch: {ms_torch:.3f}ms")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
