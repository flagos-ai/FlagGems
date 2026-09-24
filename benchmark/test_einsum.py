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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

EINSUM_SHAPES = [
    ("ij,jk->ik", ((512, 512), (512, 512))),
    ("ij,jk->ik", ((1024, 1024), (1024, 1024))),
    ("bij,bjk->bik", ((1, 512, 512), (1, 512, 512))),
    ("bij,bjk->bik", ((1, 1024, 1024), (1, 1024, 1024))),
    ("bij,bjk->bik", ((16, 512, 512), (16, 512, 512))),
    ("i,i->", ((1024,), (1024,))),
    ("i,i->", ((4096,), (4096,))),
    ("i,i->", ((65536,), (65536,))),
    ("i,j->ij", ((1024,), (1024,))),
    ("i,j->ij", ((4096,), (4096,))),
    ("ii->", ((1024, 1024),)),
    ("ii->", ((4096, 4096),)),
    ("ii->i", ((1024, 1024),)),
    ("ii->i", ((4096, 4096),)),
    ("ij->ji", ((1024, 1024),)),
    ("ij->ji", ((4096, 4096),)),
    ("ijk->", ((64, 64, 64),)),
    ("ijk->", ((128, 128, 128),)),
    ("ijk->j", ((64, 64, 64),)),
    ("ijk->j", ((128, 128, 128),)),
    ("...ij,...jk->...ik", ((2, 4, 64, 64), (2, 4, 64, 128))),
    ("...ij,...jk->...ik", ((2, 8, 128, 128), (2, 8, 128, 256))),
]


def _case_fn(shape, dtype):
    del dtype
    equation, operand_shapes = shape
    yield base.BenchmarkCasePlan(
        shape={
            f"operand{index}": list(operand_shape)
            for index, operand_shape in enumerate(operand_shapes)
        },
        params={"equation": equation},
        builder_args=(equation, operand_shapes),
    )


def _build_inputs_fn(plan, dtype, device):
    equation, operand_shapes = plan.builder_args
    operands = tuple(
        utils.generate_tensor_input(shape, dtype, device) for shape in operand_shapes
    )
    return equation, operands


class EinsumBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=EINSUM_SHAPES)


@pytest.mark.einsum
def test_einsum():
    bench = EinsumBenchmark(
        op_name="einsum",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.einsum,
        gems_op=getattr(flag_gems, "einsum", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
