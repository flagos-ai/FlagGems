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

"""Benchmark for aten::cudnn_is_acceptable.

The operator is a host-side predicate over tensor metadata, so the measured
work is dispatch plus the predicate itself.  Fixtures are uninitialized
allocations and the shared default/comprehensive shape lists are used as-is
(including their large entries, which stay metadata-only for torch.empty).
"""

import pytest
import torch

import flag_gems

from . import base, consts


def _empty_like(shape):
    """Empty counterpart of ``shape``: same rank, first extent 0.

    0-dim shapes have no empty counterpart (their numel is always 1).
    """
    if not shape:
        return None
    return (0,) + tuple(shape[1:])


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )
    empty = _empty_like(shape)
    if empty is not None:
        yield base.BenchmarkCasePlan(
            shape={"input": empty},
            params={"empty": True},
            builder_args=(empty,),
        )


def _build_inputs_fn(plan, dtype, device):
    # The predicate never reads element values, so an uninitialized allocation
    # is a valid allocation-only fixture.
    return torch.empty(plan.builder_args[0], dtype=dtype, device=device), {}


def _bench_dtypes():
    # float16/float32 (and float64 where allocatable) are accepted; bfloat16,
    # the integer types, bool and fp8 are rejected.  Both groups are covered,
    # and allocation capability comes from the static device flags.
    device = flag_gems.runtime.device
    dtypes = list(consts.FLOAT_DTYPES)
    dtypes += list(consts.INT_DTYPES) + list(consts.EXTRA_INT_DTYPES)
    dtypes += list(consts.BOOL_DTYPES)
    if not device.support_bf16:
        dtypes = [dtype for dtype in dtypes if dtype != torch.bfloat16]
    if not device.support_int64:
        dtypes = [dtype for dtype in dtypes if dtype != torch.int64]
    if device.support_fp64:
        dtypes.append(torch.float64)
    if device.support_fp8:
        dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    return dtypes


@pytest.mark.cudnn_is_acceptable
def test_cudnn_is_acceptable():
    bench = base.GenericBenchmark(
        op_name="cudnn_is_acceptable",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cudnn_is_acceptable,
        gems_op=getattr(flag_gems, "cudnn_is_acceptable", None),
        dtypes=_bench_dtypes(),
    )
    bench.run()
