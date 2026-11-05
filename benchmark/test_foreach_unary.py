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

from typing import Generator

import pytest
import torch

from flag_gems.ops._foreach_unary import UNARY_OPS

from . import base, consts


class ForeachUnaryBenchmark(base.Benchmark):
    """Benchmark for the unary `_foreach_*` family over a TensorList.

    The interesting axis for a foreach operator is the *number* of tensors, not
    just the size of one: a per-tensor implementation costs one launch each, so
    its overhead grows with list length. Each shape below is therefore expanded
    into a list of tensors, and the list length is varied alongside the shape.

    The shapes come from the `ForeachUnaryBenchmark` entry in core_shapes.yaml,
    which `base.Benchmark.set_shapes` reaches through the MRO once an operator
    name is absent from that file. Relying on the class entry rather than one
    entry per operator is deliberate: the sixty operator names here would
    otherwise each need a near-identical block, and a missing one would silently
    fall through to the shared `Benchmark` default of 1024**3 elements, which at
    sixteen tensors per call reserves 64 GiB.
    """

    # Kept modest on purpose: every shape is multiplied by LIST_LENGTH, so the
    # 4096x4096 shapes usual for single-tensor benchmarks would reserve tens of
    # gigabytes here.
    DEFAULT_SHAPES = [
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    DEFAULT_SHAPE_DESC = "M, N (per tensor, list of 16)"

    LIST_LENGTH = 16

    def set_more_shapes(self):
        more_shapes_1d = [(2**i,) for i in range(10, 20, 4)]
        more_shapes_3d = [(16, 2**i, 16) for i in range(2, 10, 4)]
        return more_shapes_1d + more_shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            tensors = [
                torch.randn(shape, device=self.device, dtype=dtype)
                for _ in range(self.LIST_LENGTH)
            ]
            yield (tensors,)


class ForeachUnaryListLengthBenchmark(ForeachUnaryBenchmark):
    """Same operator, but sweeping the TensorList length at a fixed tensor size.

    This is the axis where the shared executor is expected to pay off, and where
    a regression back to per-tensor launches would show up first.
    """

    DEFAULT_SHAPES = [(64,), (1024,), (65536,)]
    DEFAULT_SHAPE_DESC = "N (per tensor, list of 1/16/128)"

    LIST_LENGTHS = [1, 16, 128]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            for length in self.LIST_LENGTHS:
                tensors = [
                    torch.randn(shape, device=self.device, dtype=dtype)
                    for _ in range(length)
                ]
                yield (tensors,)


# Every operator the shared executor registers, so that the benchmark count
# tracks the registration count instead of drifting from it. The names are the
# table keys rather than a second hand-written list for the same reason.
BENCH_OPS = sorted(UNARY_OPS)


def _params(inplace: bool):
    """One `pytest.param` per operator, carrying that operator's own marker.

    `parametrize` alone generates the cases but attaches no marker, so
    `pytest -m foreach_abs` selects nothing and
    `benchmark/conftest.py` falls back to the node id when it derives the
    operator id for the recorded result. Attaching the marker per parameter is
    what makes each generated case addressable as its own operator; the marker
    name is the operators.yaml id, which prefixes `underscore_` for the leading
    underscore in `aten::_foreach_*`.
    """
    suffix = "_" if inplace else ""
    return [
        pytest.param(name, marks=getattr(pytest.mark, f"foreach_{name}{suffix}"))
        for name in BENCH_OPS
    ]


@pytest.mark.parametrize("name", _params(inplace=False))
def test_perf_foreach_unary(name):
    bench = ForeachUnaryBenchmark(
        op_name=f"foreach_{name}",
        torch_op=getattr(torch, f"_foreach_{name}"),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.parametrize("name", _params(inplace=True))
def test_perf_foreach_unary_(name):
    bench = ForeachUnaryBenchmark(
        op_name=f"foreach_{name}_",
        torch_op=getattr(torch, f"_foreach_{name}_"),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


# The list-length sweep is an extra axis over operators already covered above,
# so it carries the same markers rather than introducing new operator ids.
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("abs", marks=pytest.mark.foreach_abs),
        pytest.param("sin", marks=pytest.mark.foreach_sin),
    ],
)
def test_perf_foreach_unary_list_length(name):
    """The axis where the shared executor pays off, and where a regression back
    to per-tensor launches shows up first."""
    bench = ForeachUnaryListLengthBenchmark(
        op_name=f"foreach_{name}_list_length",
        torch_op=getattr(torch, f"_foreach_{name}"),
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
