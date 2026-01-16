# MASKED_SCATTER operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.masked_scatter import (
    masked_scatter as gems_masked_scatter,
)
from flag_gems.experimental_ops.masked_scatter import (
    masked_scatter_out as gems_masked_scatter_out,
)

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.performance_utils import GenericBenchmark  # noqa: E402

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


@pytest.mark.masked_scatter
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_scatter_tensor(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) > 0.75
    source = torch.randn(self.numel(), dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_mask = to_reference(mask)
    ref_source = to_reference(source)
    ref_out = torch.ops.aten.masked_scatter(ref_self, ref_mask, ref_source)

    act_self = self.clone()
    act_mask = mask.clone()
    act_source = source.clone()
    with flag_gems.use_gems():
        act_out = gems_masked_scatter(act_self, act_mask, act_source)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_scatter
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_masked_scatter_out(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) > 0.75
    source = torch.randn(self.numel(), dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(self)
    ref_mask = to_reference(mask)
    ref_source = to_reference(source)
    ref_outbuf = torch.empty_like(ref_self)
    ref_out = torch.ops.aten.masked_scatter.out(
        ref_self, ref_mask, ref_source, out=ref_outbuf
    )

    act_self = self.clone()
    act_mask = mask.clone()
    act_source = source.clone()
    act_outbuf = torch.empty_like(act_self)
    with flag_gems.use_gems():
        act_out = gems_masked_scatter_out(act_self, act_mask, act_source, act_outbuf)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.masked_scatter
def test_perf_aten_masked_scatter():
    # Define input generation logic matching the operator arguments
    def masked_scatter_input_fn(shape, dtype, device):
        self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        mask = torch.rand(shape, device=flag_gems.device) > 0.75
        source = torch.randn(self.numel(), dtype=dtype, device=flag_gems.device)
        yield self, mask, source

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=masked_scatter_input_fn,
        op_name="masked_scatter",
        torch_op=torch.ops.aten.masked_scatter,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
