# SCALAR_TENSOR operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.scalar_tensor import (  # noqa: E402
    scalar_tensor as gems_scalar_tensor,
)
from flag_gems.experimental_ops.scalar_tensor import (  # noqa: E402
    scalar_tensor_out as gems_scalar_tensor_out,
)


# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import gems_assert_close, TO_CPU
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)

@pytest.mark.scalar_tensor
@pytest.mark.parametrize("val", [-7, -1, 0, 3, 12345, -1.5, 2.75, 3.14159])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scalar_tensor_default(val, dtype):
    ref_device = "cpu" if TO_CPU else flag_gems.device
    ref_out = torch.ops.aten.scalar_tensor(
        val, dtype=dtype, device=ref_device, layout=None, pin_memory=None
    )
    with flag_gems.use_gems():
        act_out = gems_scalar_tensor(
            val, dtype=dtype, device=flag_gems.device, layout=None, pin_memory=None
        )
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.scalar_tensor
@pytest.mark.parametrize("val", [-7, -1, 0, 3, 12345, -1.5, 2.75, 3.14159])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scalar_tensor_out(val, dtype):
    ref_device = "cpu" if TO_CPU else flag_gems.device
    ref_out_buf = torch.empty((), dtype=dtype, device=ref_device)
    act_out_buf = torch.empty((), dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.scalar_tensor.out(val, out=ref_out_buf)
    with flag_gems.use_gems():
        act_out = gems_scalar_tensor_out(val, act_out_buf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.scalar_tensor
def test_perf_aten_scalar_tensor():
    # For scalar_tensor, we use a simpler approach with triton.testing.do_bench
    # since GenericBenchmark doesn't easily support keyword-only args
    quantiles = [0.5, 0.2, 0.8]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        val = 1.5
        # PyTorch reference
        ms_torch, _, _ = triton.testing.do_bench(
            lambda: torch.ops.aten.scalar_tensor(
                val, dtype=dtype, device=flag_gems.device
            ),
            rep=100,
            quantiles=quantiles,
        )

        # FlagGems implementation
        with flag_gems.use_gems():
            ms_gems, _, _ = triton.testing.do_bench(
                lambda: gems_scalar_tensor(val, dtype=dtype, device=flag_gems.device),
                rep=100,
                quantiles=quantiles,
            )

        speedup = ms_torch / ms_gems
        print(
            f"scalar_tensor {dtype}: FlagGems={ms_gems:.3f}ms, Speedup={speedup:.2f}x"
        )
