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

"""Benchmark tests for aten::is_neg.

is_neg is a metadata read of the lazy neg bit, so every case times a
constant-time Python-bool read and the shape only describes the input that
carries the flag.  Case metadata is JSON-only and builder_args stay private, so
--list-cases allocates no tensor and --case-id replay reuses the same plans as
a normal run.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils

_VARIANTS = ("plain", "neg", "double_neg")
_COMPLEX_VARIANTS = _VARIANTS + ("conj_neg",)


def _case_fn(shape, dtype):
    # conj() only sets the conjugate bit for complex dtypes.
    variants = _COMPLEX_VARIANTS if getattr(dtype, "is_complex", False) else _VARIANTS
    for variant in variants:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"variant": variant},
            builder_args=(shape, variant),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, variant = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if variant == "neg":
        inp = torch._neg_view(inp)
    elif variant == "double_neg":
        inp = torch._neg_view(torch._neg_view(inp))
    elif variant == "conj_neg":
        inp = torch._neg_view(inp.conj())
    return inp, {}


@pytest.mark.is_neg
def test_is_neg():
    bench = base.GenericBenchmark(
        op_name="is_neg",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_neg,
        gems_op=getattr(flag_gems, "is_neg", None),
        dtypes=consts.FLOAT_DTYPES + [torch.complex64],
    )
    bench.run()
