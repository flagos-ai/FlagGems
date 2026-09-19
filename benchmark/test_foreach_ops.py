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

import re
from typing import Generator

import pytest
import torch

from flag_gems.ops._foreach_binary import registered_wrappers
from flag_gems.ops._foreach_reduction import registered_wrappers as reduction_wrappers

from . import base, consts

ALL_KEYS = sorted({**registered_wrappers(), **reduction_wrappers()})

TERNARY = ("addcmul", "addcdiv")


class ForeachOpsBenchmark(base.Benchmark):
    """Benchmark for the multi-input and reducing ``_foreach_*`` operators.

    Shapes come from the ``ForeachOpsBenchmark`` entry in core_shapes.yaml.
    Keying by class rather than by operator is deliberate: there are 86
    operators here sharing one shape policy, and any name missing from that file
    would fall through the MRO to the shared ``Benchmark`` default of 1024**3
    elements, which at sixteen tensors per call reserves 64 GiB per dtype.
    """

    DEFAULT_SHAPES = [
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]
    DEFAULT_SHAPE_DESC = "M, N (per tensor, list of 16)"

    LIST_LENGTH = 16

    def __init__(self, *args, core=None, overload=None, **kwargs):
        self.core = core
        self.overload = overload
        super().__init__(*args, **kwargs)

    def set_more_shapes(self):
        return [(2**i,) for i in range(10, 20, 4)]

    def get_input_iter(self, dtype) -> Generator:
        n = self.LIST_LENGTH
        for shape in self.shapes:
            mk = lambda: [
                torch.rand(shape, device=self.device, dtype=dtype) + 1.0
                for _ in range(n)
            ]
            tensors = mk()
            if self.core in TERNARY:
                value = {
                    "Scalar": 0.5,
                    "ScalarList": [0.5] * n,
                    # ATen wants this one on the CPU.
                    "Tensor": torch.tensor([0.5] * n),
                }[self.overload]
                yield (tensors, mk(), mk(), value)
            elif self.core == "lerp":
                weight = {
                    "List": mk(),
                    "Scalar": 0.3,
                    "ScalarList": [0.3] * n,
                }[self.overload]
                yield (tensors, mk(), weight)
            elif self.core == "copy":
                yield (tensors, mk())
            elif self.core in ("max", "norm", "zero"):
                yield (tensors,)
            elif self.overload == "List":
                yield (tensors, mk())
            elif self.overload == "Scalar":
                yield (tensors, 0.5)
            elif self.overload == "ScalarList":
                yield (tensors, [0.5] * n)
            elif self.overload == "Tensor":
                yield (tensors, torch.tensor(2.0, device=self.device))
            else:
                yield (tensors,)


def _parts(key):
    base_name, _, overload = key.partition(".")
    name = base_name[len("_foreach_") :]
    inplace = name.endswith("_")
    return (name[:-1] if inplace else name), overload, inplace


def _marker_name(key):
    """The operators.yaml id for a key, which is also its pytest marker."""
    core, overload, inplace = _parts(key)
    ident = f"foreach_{core}"
    if overload:
        ident += "_" + re.sub(r"(?<!^)(?=[A-Z])", "_", overload).lower()
    if inplace:
        ident += "_"
    return ident


def _torch_op(key):
    base_name, _, overload = key.partition(".")
    op = getattr(torch.ops.aten, base_name)
    return getattr(op, overload) if overload else op


BENCH_KEYS = [k for k in ALL_KEYS if not k.endswith(".ScalarAndTensor")]

PARAMS = [
    pytest.param(k, marks=getattr(pytest.mark, _marker_name(k))) for k in BENCH_KEYS
]


@pytest.mark.parametrize("key", PARAMS)
def test_perf_foreach_ops(key):
    core, overload, inplace = _parts(key)
    bench = ForeachOpsBenchmark(
        op_name=_marker_name(key),
        torch_op=_torch_op(key),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=inplace,
        core=core,
        overload=overload,
    )
    bench.run()


class ForeachPowScalarAndTensorBenchmark(ForeachOpsBenchmark):
    """``pow.ScalarAndTensor`` takes the scalar base first and the list second.

    The argument order is reversed relative to every other overload, so it needs
    its own iterator rather than a branch in the shared one.
    """

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            tensors = [
                torch.rand(shape, device=self.device, dtype=dtype) + 1.0
                for _ in range(self.LIST_LENGTH)
            ]
            yield (2.0, tensors)


@pytest.mark.foreach_pow_scalar_and_tensor
def test_perf_foreach_pow_scalar_and_tensor():
    bench = ForeachPowScalarAndTensorBenchmark(
        op_name="foreach_pow_scalar_and_tensor",
        torch_op=torch.ops.aten._foreach_pow.ScalarAndTensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
