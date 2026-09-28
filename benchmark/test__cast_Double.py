# Copyright 2025 The FlagGems Authors.
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

"""Benchmark for ``torch.ops.aten._cast_Double``.

``GenericBenchmark`` keeps the original core/comprehensive family, including its
1<<30 element case and its own comprehensive expansions; the spec value-grid
shapes and the 0-dim scalar / zero-extent call forms are appended to those
expansions, with nothing removed, shrunk or capped. Every result is float64, so
the run follows the backend's static fp64 capability flag.
"""

import numbers

import pytest
import torch

import flag_gems

from . import base, utils


def _bench_dtypes():
    support = flag_gems.runtime.device
    if not support.support_fp64:
        # Every result is float64: without fp64 support no workload is valid.
        return []
    dtypes = [torch.float32, torch.float16, torch.float64, torch.int32]
    if support.support_bf16:
        dtypes.append(torch.bfloat16)
    return dtypes


# Appended to the comprehensive expansions: the spec value-grid shapes plus the
# 0-dim scalar and zero-extent call forms. A caller shape file still determines
# this operator's geometry whenever it lists them.
_EXTRA_SHAPES = [
    (2**24,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (),
    (0, 3),
]

# None marks the omitted-argument call form. It stays in the private
# builder_args only; the published params metadata carries the explicit bool
# forms, so listing shows two flag values and an empty mapping for omission.
_NON_BLOCKING = (None, False, True)


def _validated_shape(shape):
    """Reject malformed extents before any benchmark plan is built."""
    if isinstance(shape, numbers.Integral) and not isinstance(shape, bool):
        shape = (int(shape),)
    if isinstance(shape, (str, bytes)) or not isinstance(shape, (tuple, list)):
        raise ValueError(f"shape must be a sequence of extents, got {shape!r}")
    extents = []
    for extent in shape:
        # bool is an int subclass, so a True/False extent is a caller mistake.
        if isinstance(extent, bool) or not isinstance(extent, numbers.Integral):
            raise ValueError(f"shape extent must be an integer, got {extent!r}")
        if extent < 0:
            raise ValueError(f"shape extent must be non-negative, got {extent}")
        extents.append(int(extent))
    return tuple(extents)


def _case_fn(shape, dtype):
    del dtype
    shape = _validated_shape(shape)
    plan_shape = {"input": shape}
    for non_blocking in _NON_BLOCKING:
        params = {} if non_blocking is None else {"non_blocking": non_blocking}
        yield base.BenchmarkCasePlan(
            shape=plan_shape,
            params=params,
            builder_args=(shape, non_blocking),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    if dtype == torch.float64:
        # The shared generator covers the common dtype tables only and would
        # hand back None for float64, so the pass-through input is built
        # directly in its own precision rather than staged through float32.
        inp = torch.randn(shape, dtype=torch.float64, device=device)
    else:
        inp = utils.generate_tensor_input(shape, dtype, device)
    if non_blocking is None:
        return inp, {}
    return inp, {"non_blocking": non_blocking}


class _CastDoubleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # The generic comprehensive expansions plus the spec shapes. Base
        # set_shapes still honours a caller shape file, keeps that file's
        # operator/class precedence and rejects a nonexistent explicit path.
        return list(super().set_more_shapes()) + list(_EXTRA_SHAPES)


@pytest.mark.cast_Double
def test__cast_Double():
    bench = _CastDoubleBenchmark(
        op_name="_cast_Double",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cast_Double,
        gems_op=getattr(flag_gems, "_cast_Double", None),
        dtypes=_bench_dtypes(),
    )
    bench.run()
