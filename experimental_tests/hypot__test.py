# HYPOT_ operator test

import os
import sys

import pytest
import torch
import triton  # noqa: F401

import flag_gems
from flag_gems.experimental_ops.hypot_ import hypot_ as gems_hypot_

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from benchmark.performance_utils import GenericBenchmark
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


@pytest.mark.hypot_
@pytest.mark.parametrize(
    "self_shape,other_shape",
    [
        ((2, 3), (2, 3)),
        ((2, 3), (1, 3)),
        ((2, 3), (2, 1)),
        ((2, 3), (1, 1)),
        ((128, 256), (128, 256)),
        ((128, 256), (1, 256)),
        ((128, 256), (128, 1)),
        ((128, 256), (1, 1)),
        ((512, 512), (512, 512)),
        ((512, 512), (1, 512)),
        ((512, 512), (512, 1)),
        ((512, 512), (1, 1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("contig", [True, False])
def test_hypot__tensor(self_shape, other_shape, dtype, contig):
    if contig:
        base_self = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    else:
        src = torch.randn(
            (self_shape[1], self_shape[0]), dtype=dtype, device=flag_gems.device
        )
        base_self = src.permute(1, 0)
    base_other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)

    ref_self = to_reference(base_self)
    ref_other = to_reference(base_other)
    ref_out = torch.ops.aten.hypot_(ref_self, ref_other)

    act_self = base_self.clone()
    act_other = base_other.clone()
    with flag_gems.use_gems():
        act_out = gems_hypot_(act_self, act_other)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.hypot_
def test_perf_aten_hypot_():
    # Define input generation logic matching the operator arguments
    def hypot__input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=hypot__input_fn,
        op_name="hypot_",
        torch_op=torch.ops.aten.hypot_,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
