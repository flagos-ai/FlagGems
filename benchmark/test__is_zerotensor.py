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

"""Benchmark for ``aten::_is_zerotensor``.

A host-side metadata query, so the headline timing reflects dispatch overhead.
Every shape is measured on both the flagged (``_efficientzerotensor``) and the
ordinary dense-zero operand, keeping the marker path distinct from a dense
fallback. Core uses the shared ``consts.DEFAULT_SHAPES``; the comprehensive
level adds the framework extras plus the operator's original shapes.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# Merged only at the comprehensive level (see base.Benchmark.set_shapes); the
# core level keeps the shared defaults.
_MORE_SHAPES = [
    (1024,),
    (64, 64),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
]

_KINDS = ("zerotensor", "dense_zero")

# Include both supported FP8 storage types for the metadata query.
_BENCH_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + consts.BOOL_DTYPES
    + consts.COMPLEX_DTYPES
    + (
        [torch.float8_e4m3fn, torch.float8_e5m2]
        if flag_gems.runtime.device.support_fp8
        else []
    )
)


def _case_fn(shape, dtype):
    del dtype
    for kind in _KINDS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"kind": kind},
            builder_args=(shape, kind),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, kind = plan.builder_args
    if kind == "zerotensor":
        # The only public ZeroTensor constructor; it allocates no storage.
        inp = torch.ops.aten._efficientzerotensor(shape, dtype=dtype, device=device)
    else:
        # Same all-zero values as the flagged operand, but an ordinary tensor
        # whose marker is unset.
        inp = torch.zeros(shape, dtype=dtype, device=device)
    return (inp,)


class _IsZeroTensorBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return super().set_more_shapes() + _MORE_SHAPES


@pytest.mark.is_zerotensor
def test__is_zerotensor():
    bench = _IsZeroTensorBenchmark(
        op_name="_is_zerotensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._is_zerotensor,
        gems_op=getattr(flag_gems, "_is_zerotensor", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
