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

# aten::size is a host-side metadata query, so the measured work is the dispatch
# plus the attribute read and the payload is never touched. The shapes only need
# to be realistic and cheap to allocate.
SIZE_SHAPES = [
    (),
    (256,),
    (2, 19, 7),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


def _case_fn(shape, dtype):
    del dtype
    # Plans hold JSON metadata only; _build_inputs_fn allocates the tensor, so
    # --list-cases stays tensor-free.
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"dim": None},
        builder_args=(shape, None),
    )
    if shape:
        # int call form, at both boundary positions of the dim argument.
        for dim in (0, -1):
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={"dim": dim},
                builder_args=(shape, dim),
            )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if dim is None:
        return inp, {}
    return inp, dim, {}


class SizeBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SIZE_SHAPES)


@pytest.mark.size
def test_size():
    bench = SizeBenchmark(
        op_name="size",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.size,
        gems_op=getattr(flag_gems, "size", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
