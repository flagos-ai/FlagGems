import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

PUT_BENCH_SHAPES = [(16,), (64, 64), (20, 320, 15), (16, 128, 64, 60)]
PUT_MAX_NUMEL = 2**28


class PutBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)
        self.shapes = [s for s in self.shapes if math.prod(s) < PUT_MAX_NUMEL]

    def set_more_shapes(self):
        return PUT_BENCH_SHAPES


def _case_fn_for(accumulate):
    def case_fn(shape, dtype):
        del dtype
        dims = tuple(shape)
        count = max(1, min(math.prod(dims) // 2, 2**20))
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "index": [count], "source": [count]},
            params={"accumulate": accumulate},
            builder_args=(dims, count),
        )

    return case_fn


def _put_randperm(numel, *, device):
    # Enflame's native int64 randperm aborts before the operator under test.
    # Build the same unique-index fixture on CPU, then preserve its target ABI.
    factory_device = "cpu" if flag_gems.vendor_name == "enflame" else device
    return torch.randperm(numel, device=factory_device).to(device)


def _build_inputs_fn(plan, dtype, device):
    shape, count = plan.builder_args
    if (
        flag_gems.vendor_name == "ascend"
        and dtype == torch.bfloat16
        and plan.params["accumulate"]
    ):
        pytest.skip("Ascend native put_ accumulate does not support BFloat16")
    inp = torch.randn(shape, dtype=dtype, device=device)
    numel = inp.numel()
    if plan.params["accumulate"]:
        index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
    else:
        index = _put_randperm(numel, device=device)[:count].to(torch.int64)
    source = torch.randn(count, dtype=dtype, device=device)
    return inp, index, source, plan.params["accumulate"]


@pytest.mark.put_
def test_put_():
    PutBenchmark(
        op_name="put_",
        torch_op=torch.ops.aten.put_,
        case_fn=_case_fn_for(False),
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()


@pytest.mark.put_
def test_put__accumulate():
    PutBenchmark(
        op_name="put_",
        torch_op=torch.ops.aten.put_,
        case_fn=_case_fn_for(True),
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    ).run()
