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

import math

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
# only narrows the flat buffer and views each chunk, so a case is a 1-D flat
# buffer plus the donor shapes whose element counts total the flat size. The
# cases mirror benchmark/test_flatten_dense_tensors.py (1M - 16.8M elements,
# 1 to 1024 donors).
UNFLATTEN_DENSE_TENSORS_SHAPES = [
    (1024 * 1024, [(1024, 1024)]),
    (4096 * 4096, [(4096, 4096)]),
    (3 * 1024 * 1024, [(1024, 1024), (1024, 1024), (1024, 1024)]),
    (64 * 512 * 512, [(64, 512, 512)]),
    (2 * 2048 * 2048, [(2048, 2048), (2048, 2048)]),
    (3 * 16384 * 256, [(16384, 256), (16384, 256), (16384, 256)]),
    (1024 * 1024, [(1024,)] * 1024),
]


def _case_fn(shape, dtype):
    del dtype
    numel, donor_shapes = shape
    yield base.BenchmarkCasePlan(
        shape={"flat": (numel,), "donors": donor_shapes},
        params={"numel": numel, "donor_count": len(donor_shapes)},
        builder_args=(numel, donor_shapes),
    )


def _build_inputs_fn(plan, dtype, device):
    numel, donor_shapes = plan.builder_args
    flat = utils.generate_tensor_input((numel,), dtype, device)
    # Donors are views of the flat buffer: only their shapes are read, so this
    # adds no allocation. Both arguments are returned positionally, matching the
    # op(flat, tensors) call form of the reference and of the candidate.
    donors, offset = [], 0
    for shape in donor_shapes:
        size = torch.Size(shape).numel()
        donors.append(flat[offset : offset + size].view(shape))
        offset += size
    return flat, donors


class UnflattenDenseTensorsBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to (flat, donor-shapes) cases."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        rows = []
        for shape in list(self.shapes) + UNFLATTEN_DENSE_TENSORS_SHAPES:
            if len(shape) == 2 and isinstance(shape[1], (tuple, list)):
                row = (int(shape[0]), tuple(tuple(donor) for donor in shape[1]))
            else:
                row = (math.prod(shape), (tuple(shape),))
            if row not in rows:
                rows.append(row)
        self.shapes = rows


@pytest.mark.unflatten_dense_tensors
def test_unflatten_dense_tensors():
    bench = UnflattenDenseTensorsBenchmark(
        op_name="unflatten_dense_tensors",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.unflatten_dense_tensors,
        gems_op=getattr(flag_gems, "unflatten_dense_tensors", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
