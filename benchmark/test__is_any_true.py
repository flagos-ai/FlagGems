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

"""Benchmark for ``aten::_is_any_true``.

The operator is a whole-tensor boolean scan producing a 0-dim bool tensor, so
``torch_op`` (the perf comparison reference) and ``gems_op`` (the candidate) are
called identically as ``op(input)``. No public Benchmark family covers a
single-operand reduction to a 0-dim bool, so this file uses the two-phase
``case_fn``/``build_inputs_fn`` API: one plan set feeds both ``--list-cases`` and
normal execution.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# ``_is_any_true`` only accepts bool tensors on the probed backend, so bool is
# the only benchmarked dtype. A --shape_file entry for "_is_any_true" (or for
# IsAnyTrueBenchmark) replaces this list wholesale; no shape given that way is
# filtered out here.
IS_ANY_TRUE_SHAPES = [
    (65536,),
    (1048576,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
    (1,),
    (),
    (0,),
    (2, 0, 4),
]

# Each pattern is built deterministically by the builder below, so the case
# metadata always describes the tensor the builder produces. all_false makes
# every element False, while early_true and tail_true place the only True at
# opposite ends of the flattened input; no assumption is made about how a
# candidate traverses its input. An empty tensor cannot hold a True and so only
# gets all_false.
_PATTERNS = ("all_false", "early_true", "tail_true")


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _patterns_for(shape):
    return _PATTERNS if _numel(shape) else ("all_false",)


def _case_fn(shape, dtype):
    del dtype
    for pattern in _patterns_for(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"pattern": pattern},
            builder_args=(shape, pattern),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, pattern = plan.builder_args
    inp = torch.zeros(shape, dtype=dtype, device=device)
    if pattern != "all_false":
        inp.view(-1)[0 if pattern == "early_true" else -1] = True
    return inp, {}


def _normalize_shape(shape):
    """Return one normalized shape: a tuple of validated non-negative extents.

    A bare integer entry is normalized to its 1-D tuple here, because the
    inherited ``OperatorBenchmark.as_tuple`` leaves scalar shape entries
    unchanged and a downstream plan would then iterate an int. ``bool`` is a
    subclass of ``int`` and is excluded explicitly so ``True`` cannot pass as
    ``1``; negative and non-integer extents are rejected before allocation.
    The normalized value is what the case plans carry, so listing and
    ``build_inputs_fn`` always agree.
    """
    dims = shape if isinstance(shape, (tuple, list)) else (shape,)
    normalized = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"_is_any_true: invalid shape extent in {shape!r}")
        normalized.append(int(dim))
    return tuple(normalized)


class IsAnyTrueBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with a scan-oriented shape list."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=IS_ANY_TRUE_SHAPES)
        self.shapes = [_normalize_shape(shape) for shape in self.shapes]


@pytest.mark.is_any_true
def test__is_any_true():
    bench = IsAnyTrueBenchmark(
        op_name="_is_any_true",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._is_any_true,
        gems_op=getattr(flag_gems, "_is_any_true", None),
        dtypes=consts.BOOL_DTYPES,
    )
    bench.run()
