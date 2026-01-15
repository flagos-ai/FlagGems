# _FUNCTIONAL_SYM_CONSTRAIN_RANGE_FOR_SIZE operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops._functional_sym_constrain_range_for_size import (
    _functional_sym_constrain_range_for_size as gems__functional_sym_constrain_range_for_size,
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


@pytest.mark.functional_sym_constrain_range_for_size
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        {"size": 5, "min": None, "max": None},
        {"size": 10, "min": 0, "max": None},
        {"size": 0, "min": 0, "max": 1000},
        {"size": 128, "min": 1, "max": 1024},
        {"size": 255, "min": 0, "max": 256},
        {"size": 1, "min": 1, "max": 10},
        {"size": 32, "min": None, "max": 64},
        {"size": 64, "min": 32, "max": None},
    ],
)
def test__functional_sym_constrain_range_for_size_scalar_token(shape, dtype, case):
    dep_token = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    dep_token_ref = to_reference(dep_token)
    dep_token_act = dep_token.clone()

    size = case["size"]
    min_val = case["min"]
    max_val = case["max"]

    ref_out = torch.ops.aten._functional_sym_constrain_range_for_size(
        size, min_val, max_val, dep_token_ref
    )

    with flag_gems.use_gems():
        act_out = gems__functional_sym_constrain_range_for_size(
            size, min_val, max_val, dep_token_act
        )

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.functional_sym_constrain_range_for_size
def test_perf_aten__functional_sym_constrain_range_for_size():
    # Define input generation logic matching the operator arguments
    def _functional_sym_constrain_range_for_size_input_fn(shape, dtype, device):
        dep_token = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        size = 5  # Example size, can be varied
        min_val = None  # Example min, can be varied
        max_val = None  # Example max, can be varied
        yield size, min_val, max_val, dep_token

    # Initialize benchmark
    bench = GenericBenchmark(
        input_fn=_functional_sym_constrain_range_for_size_input_fn,
        op_name="_functional_sym_constrain_range_for_size",
        torch_op=torch.ops.aten._functional_sym_constrain_range_for_size,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()
