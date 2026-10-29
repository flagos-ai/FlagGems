# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

DEFAULT_SHAPES = [(16,), (64, 64), (20, 320, 15), (16, 128, 64, 60)]
MAX_NUMEL = 2**28


class PutBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_SHAPES)
        self.shapes = [shape for shape in self.shapes if math.prod(shape) < MAX_NUMEL]


def _case_fn_factory(accumulate, use_out=False):
    def _case_fn(shape, dtype):
        del dtype
        dims = tuple(shape)
        count = max(1, min(math.prod(dims) // 2, 2**20))
        shape_meta = {"self": list(dims), "index": [count], "source": [count]}
        if use_out:
            shape_meta["out"] = list(dims)
        yield base.BenchmarkCasePlan(
            shape=shape_meta,
            params={"accumulate": accumulate, "out": use_out},
            builder_args=(dims, count, accumulate, use_out),
        )

    return _case_fn


def _put_randperm(numel, *, device):
    # Enflame's native int64 randperm aborts before the operator under test.
    # Build the same unique-index fixture on CPU, then preserve its target ABI.
    factory_device = "cpu" if flag_gems.vendor_name == "enflame" else device
    return torch.randperm(numel, device=factory_device).to(device)


def _build_inputs_fn(plan, dtype, device):
    shape, count, accumulate, use_out = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    index = (
        torch.randint(0, inp.numel(), (count,), dtype=torch.int64, device=device)
        if accumulate
        else _put_randperm(inp.numel(), device=device)[:count].to(torch.int64)
    )
    source = torch.randn((count,), dtype=dtype, device=device)
    if use_out:
        return inp, index, source, accumulate, {"out": torch.empty_like(inp)}
    return inp, index, source, {"accumulate": accumulate}


@pytest.mark.put
@pytest.mark.parametrize("accumulate", (False, True))
def test_put(accumulate):
    bench = PutBenchmark(
        op_name="put",
        torch_op=torch.put,
        case_fn=_case_fn_factory(accumulate),
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# Preserve the consolidated legacy put.out benchmark family.
@pytest.mark.put_out
@pytest.mark.parametrize("accumulate", (False, True))
def test_put_out_shared_family(accumulate):
    bench = PutBenchmark(
        op_name="put_out",
        torch_op=torch.ops.aten.put.out,
        case_fn=_case_fn_factory(accumulate, use_out=True),
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
