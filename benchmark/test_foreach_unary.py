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

from . import base


class ForeachUnaryBenchmark(base.Benchmark):
    """Benchmark for the unary `_foreach_*` family over a TensorList.

    The interesting axis for a foreach operator is the *number* of tensors, not
    just the size of one: a per-tensor implementation costs one launch each, so
    its overhead grows with list length. Each shape below is therefore expanded
    into a list of tensors, and the list length is varied alongside the shape.
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


BENCH_OPS = ["abs", "neg", "sin", "exp", "sqrt", "round", "sigmoid"]


@pytest.mark.parametrize("name", BENCH_OPS)
def test_perf_foreach_unary(name):
    bench = ForeachUnaryBenchmark(
        op_name=f"foreach_{name}",
        torch_op=getattr(torch, f"_foreach_{name}"),
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()


@pytest.mark.parametrize("name", BENCH_OPS)
def test_perf_foreach_unary_(name):
    bench = ForeachUnaryBenchmark(
        op_name=f"foreach_{name}_",
        torch_op=getattr(torch, f"_foreach_{name}_"),
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.run()


@pytest.mark.parametrize("name", ["abs", "sin"])
def test_perf_foreach_unary_list_length(name):
    """The axis where the shared executor pays off, and where a regression back
    to per-tensor launches shows up first."""
    bench = ForeachUnaryListLengthBenchmark(
        op_name=f"foreach_{name}_list_length",
        torch_op=getattr(torch, f"_foreach_{name}"),
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
