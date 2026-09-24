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

"""Benchmark for ``aten::diagflat(Tensor self, int offset=0) -> Tensor``.

``diagflat`` flattens the input (logical row-major) and writes it onto the
diagonal of a square matrix with side length ``numel(input) + |offset|``, so the
output is quadratic in the input element count. The default shape collection
contains huge tensors whose quadratic output would exhaust device memory, so the
two-phase :class:`GenericBenchmark` is restricted to the bounded shapes below
(~4096 input elements -> ~16M output elements, the largest case).
"""

import pytest
import torch

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

DIAGFLAT_SHAPES = [
    (64,),
    (256,),
    (1024,),
    (2048,),
    (4096,),
    (32, 32),
    (64, 64),
    (16, 16, 16),
]

# offsets exercised per shape: the main diagonal plus a shifted one (the output
# side grows by |offset|).
DIAGFLAT_OFFSETS = [0, 3]


def _case_fn(shape, dtype):
    del dtype
    for offset in DIAGFLAT_OFFSETS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"offset": offset},
            builder_args=(shape, offset),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, offset = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"offset": offset}


class DiagFlatBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to bounded input shapes.

    The default shape levels would make the quadratic diagflat output explode,
    so the case list is restricted to the small shapes above and the extra
    comprehensive shapes are disabled.
    """

    DEFAULT_SHAPE_DESC = "input numel (output side = numel + |offset|)"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DIAGFLAT_SHAPES)

    def set_more_shapes(self):
        return []


@pytest.mark.diagflat
def test_diagflat():
    bench = DiagFlatBenchmark(
        op_name="diagflat",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.diagflat,
        # KernelGen injects the candidate via --override; the default
        # module callable may not exist until the op is merged into FlagGems.
        gems_op=getattr(flag_gems, "diagflat", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
