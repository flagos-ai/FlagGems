# FFT_IFFTSHIFT operator test

import os
import sys

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402, F401

import flag_gems  # noqa: E402
from flag_gems.experimental_ops.fft_ifftshift import (  # noqa: E402
    fft_ifftshift as gems_fft_ifftshift,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.fft_ifftshift
@pytest.mark.parametrize(
    "shape", [(2, 3), (128, 256), (512, 512), (1024,), (16, 17, 18)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [None, [0], [-1]])
def test_fft_ifftshift_tensor(shape, dtype, dim):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_input = to_reference(input_tensor)
    ref_out = torch.ops.aten.fft_ifftshift(ref_input, dim)
    with flag_gems.use_gems():
        act_out = gems_fft_ifftshift(input_tensor, dim)
    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.fft_ifftshift
def test_perf_aten_fft_ifftshift():
    # Define input generation logic matching the operator arguments
    def fft_ifftshift_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp,

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=fft_ifftshift_input_fn,
        op_name="fft_ifftshift",
        torch_op=torch.ops.aten.fft_ifftshift,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
