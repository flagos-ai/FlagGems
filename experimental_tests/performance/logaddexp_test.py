import os
import sys

import pytest
import torch
import time
import flag_gems
import triton


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
try:
    from benchmark.performance_utils import GenericBenchmark
    from tests.accuracy_utils import TO_CPU, gems_assert_close

except ImportError:
    TO_CPU = False

    def gems_assert_close(res, ref, dtype, **kwargs):
        torch.testing.assert_close(res, ref, **kwargs)
        
def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.close()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp



@pytest.mark.logaddexp
def test_perf_aten_logaddexp():
    def logaddexp_input_fn(shape, dtype, device):
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        yield inp1, inp2

    bench = GenericBenchmark(
        input_fn = logaddexp_input_fn,
        op_name = "logaddexp",
        torch_op = torch.ops.aten.logaddexp,
        dtypes = [torch.float32, torch.float16, torch.bfloat16],
    )

    return bench.run()

