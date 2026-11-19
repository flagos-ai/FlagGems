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

import pytest
import torch

import flag_gems

from . import base, consts


def logcumsumexp_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp, 1


@pytest.mark.logcumsumexp
def test_logcumsumexp():
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp",
        input_fn=logcumsumexp_input_fn,
        torch_op=torch.logcumsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def logcumsumexp_out_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty_like(inp)
    yield inp, 1, {"out": out}


@pytest.mark.logcumsumexp_backward
def test_logcumsumexp_backward():
    # ``use_gems()`` keeps the ATen autograd node above the device kernel, so
    # the reference backward would still win on the ``torch.*`` path. Benchmark
    # the public FlagGems entry directly to measure the Triton backward kernel.
    #
    # fp16 is deliberately absent: torch 2.8 raises
    # ``"logcumsumexp_backward" not implemented for 'Half'`` for the native
    # baseline (the candidate itself supports fp16 and is covered by
    # tests/test_logcumsumexp.py). fp32 and bf16 exercise both the fp32 and the
    # bf16 kernel paths.
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp",
        input_fn=logcumsumexp_input_fn,
        torch_op=torch.logcumsumexp,
        dtypes=[torch.float32, torch.bfloat16],
        is_backward=True,
    )
    bench.set_gems(flag_gems.logcumsumexp)
    bench.run()


@pytest.mark.logcumsumexp_out
def test_logcumsumexp_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="logcumsumexp_out",
        input_fn=logcumsumexp_out_input_fn,
        torch_op=torch.ops.aten.logcumsumexp.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
