# RESHAPE operator test

import os
import sys

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
try:
    from tests.accuracy_utils import gems_assert_close
except ImportError:
    # Fallback values when running outside pytest

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.reshape import reshape as gems_reshape  # noqa: E402


@pytest.mark.reshape
@pytest.mark.parametrize(
    "in_shape,out_shape",
    [
        ((2, 3), (3, 2)),
        ((2, 3), (1, 6)),
        ((2, 3), (-1,)),
        ((128, 256), (256, 128)),
        ((128, 256), (32, -1)),
        ((128, 256), (-1, 128)),
        ((64, 64, 64), (-1,)),
        ((64, 64, 64), (64, 4096)),
        ((512, 512), (-1,)),
        ((256, 512), (512, 256)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reshape_tensor_contiguous(in_shape, out_shape, dtype):
    input_tensor = torch.randn(in_shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    ref_out = torch.ops.aten.reshape(ref_input, out_shape)

    with flag_gems.use_gems():
        act_out = gems_reshape(input_tensor, out_shape)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reshape
@pytest.mark.parametrize(
    "base_shape,out_shape,transform",
    [
        ((32, 64), (-1,), "transpose01"),
        ((64, 128), (128, 64), "transpose01"),
        ((8, 16, 32), (256, 16), "permute201"),
        ((64, 64, 64), (4096, 64), "permute120"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reshape_tensor_noncontiguous(base_shape, out_shape, transform, dtype):
    base = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    if transform == "transpose01":
        ref_input = ref_input.transpose(0, 1)
        act_input = act_input.transpose(0, 1)
    elif transform == "permute201":
        ref_input = ref_input.permute(2, 0, 1)
        act_input = act_input.permute(2, 0, 1)
    elif transform == "permute120":
        ref_input = ref_input.permute(1, 2, 0)
        act_input = act_input.permute(1, 2, 0)

    ref_out = torch.ops.aten.reshape(ref_input, out_shape)

    with flag_gems.use_gems():
        act_out = gems_reshape(act_input, out_shape)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.reshape
@pytest.mark.parametrize(
    "in_shape,out_shape",
    [
        ((2, 3), (3, 2)),
        ((2, 3), (1, 6)),
        ((2, 3), (-1,)),
        ((128, 256), (256, 128)),
        ((128, 256), (32, -1)),
        ((128, 256), (-1, 128)),
        ((64, 64, 64), (-1,)),
        ((64, 64, 64), (64, 4096)),
        ((512, 512), (-1,)),
        ((256, 512), (512, 256)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reshape_tensor_contiguous_performance(in_shape, out_shape, dtype):
    import torch.utils.benchmark as benchmark  # noqa: E402, F401

    quantiles = [0.5, 0.2, 0.8]

    input_tensor = torch.randn(in_shape, dtype=dtype, device=flag_gems.device)
    ref_input = input_tensor.clone()

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reshape(ref_input, out_shape),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reshape(input_tensor, out_shape), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reshape {out_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.reshape
@pytest.mark.parametrize(
    "base_shape,out_shape,transform",
    [
        ((32, 64), (-1,), "transpose01"),
        ((64, 128), (128, 64), "transpose01"),
        ((8, 16, 32), (256, 16), "permute201"),
        ((64, 64, 64), (4096, 64), "permute120"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reshape_tensor_noncontiguous_performance(
    base_shape, out_shape, transform, dtype
):
    quantiles = [0.5, 0.2, 0.8]

    base = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    ref_input = base.clone()
    act_input = base.clone()

    if transform == "transpose01":
        ref_input = ref_input.transpose(0, 1)
        act_input = act_input.transpose(0, 1)
    elif transform == "permute201":
        ref_input = ref_input.permute(2, 0, 1)
        act_input = act_input.permute(2, 0, 1)
    elif transform == "permute120":
        ref_input = ref_input.permute(1, 2, 0)
        act_input = act_input.permute(1, 2, 0)

    # PyTorch reference implementation
    ms_torch, _, _ = triton.testing.do_bench(
        lambda: torch.ops.aten.reshape(ref_input, out_shape),
        rep=100,
        quantiles=quantiles,
    )

    # Triton implementation
    with flag_gems.use_gems():
        ms_triton, _, _ = triton.testing.do_bench(
            lambda: gems_reshape(act_input, out_shape), rep=100, quantiles=quantiles
        )

    # Calculate speedup and return result
    speedup = ms_torch / ms_triton

    print(f"reshape {out_shape} {dtype}:")
    print(f"  FlagGems: {ms_triton:.3f}ms")
    print(f"  Speedup: {speedup:.2f}x")
