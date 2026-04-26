import pytest
import torch

import flag_gems

from . import performance_utils as base


def _input_fn(shape, dtype, device):
    yield {"n": shape[0], "dtype": dtype, "device": device},


@pytest.mark.randperm
def test_randperm(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("DISABLE_LLVM_OPT", "1")

    bench = base.GenericBenchmark(
        op_name="randperm",
        input_fn=_input_fn,
        torch_op=torch.randperm,
        dtypes=[torch.int32, torch.int64],
    )
    bench.run()
