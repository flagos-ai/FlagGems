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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The operator is a dtype-agnostic storage relocation, so every supported dtype
# family is timed: float, complex, int and bool.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.COMPLEX_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if _dtype_supported(dtype)
]


# (target shape, donor storage shape, donor view kind). set_data only swaps
# storage and type metadata, so the measured work is the replacement itself; the
# rows vary rank and element count (the target may grow or shrink) and include a
# transposed donor, which hands the target non-contiguous strides.
BENCH_SHAPES = [
    ([64, 64], [64, 64], "as_is"),
    ([1024, 1024], [1024, 1024], "as_is"),
    ([20, 320, 15], [20, 320, 15], "as_is"),
    ([1024, 1024], [20, 320, 15], "as_is"),
    ([20, 320, 15], [1024, 1024], "as_is"),
    ([16, 128, 64, 60], [16, 7, 57, 32, 29], "as_is"),
    ([16, 7, 57, 32, 29], [16, 128, 64, 60], "as_is"),
    ([1024, 1024], [1024, 1024], "transposed"),
]


def _case_fn(shape, dtype):
    # Each row is a (target, donor storage, donor view) descriptor, so one plan
    # per row keeps both shapes in the case list while the tensors stay unbuilt.
    del dtype
    target_shape, storage_shape, kind = shape
    yield base.BenchmarkCasePlan(
        shape={"target": list(target_shape), "source": list(storage_shape)},
        params={"source_layout": kind},
        builder_args=(target_shape, storage_shape, kind),
    )


def _build_inputs_fn(plan, dtype, device):
    target_shape, storage_shape, kind = plan.builder_args
    target = torch.empty(target_shape, dtype=dtype, device=device)
    storage = torch.empty(storage_shape, dtype=dtype, device=device)
    source = (
        storage.transpose(0, 1)
        if kind == "transposed" and storage.dim() > 1
        else storage
    )
    return target, source


class SetDataBenchmark(OperatorBenchmark):
    """State-changing op, so every sample needs an independent target/donor pair.

    ``fresh_inputs`` restores that pair outside the measured region; without it a
    repeated call would only re-apply a swap the previous sample already did. The
    shared helper restores with ``detach().clone().requires_grad_(...)``, so
    autograd history and cross-input aliases are not preserved - nothing here
    relies on them, the rows only carry shapes and a donor layout kind.

    Shared shapes and the original rank-changing donor cases are both retained.
    """

    DEFAULT_SHAPE_DESC = "set_data metadata-swap cases"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # A caller file may list bare shapes or (target, donor) pairs; normalize
        # them to the descriptor form instead of dropping valid workloads.
        rows = []
        for row in list(self.shapes) + BENCH_SHAPES:
            if all(isinstance(item, int) for item in row):
                rows.append((tuple(row), tuple(row), "as_is"))
            elif len(row) == 2:
                rows.append((tuple(row[0]), tuple(row[1]), "as_is"))
            else:
                rows.append((tuple(row[0]), tuple(row[1]), row[2]))
        self.shapes = list(dict.fromkeys(rows))


@pytest.mark.set_data
def test_set_data():
    bench = SetDataBenchmark(
        op_name="set_data",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.set_data,
        gems_op=getattr(flag_gems, "set_data", None),
        dtypes=BENCH_DTYPES,
        fresh_inputs=True,
    )
    bench.run()
