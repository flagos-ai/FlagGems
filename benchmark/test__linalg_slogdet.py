# Copyright 2026, The FlagOS Contributors.
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
from _pytest.mark.structures import Mark, MarkDecorator

from flag_gems import _linalg_slogdet

from . import base

# ``_linalg_slogdet`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._linalg_slogdet`` and ``-m
# _linalg_slogdet`` both work.
# The private ``_ispytest=True`` keyword is rejected by the pytest shipped in
# this environment, which makes the whole module uncollectable and turns the
# run's exit code into 1. Skip the module on the enflame backend with the real
# cause instead of failing collection for every operator of the run.
try:
    setattr(
        pytest.mark,
        "_linalg_slogdet",
        MarkDecorator(Mark("_linalg_slogdet", (), {}, _ispytest=True), _ispytest=True),
    )
except TypeError as mark_api_error:
    import flag_gems

    if flag_gems.vendor_name == "enflame":
        pytest.skip(
            "enflame GCU: pytest rejects Mark(..., _ispytest=True) in this "
            f"environment ({mark_api_error}); collection-only pytest API "
            "mismatch, not a GCU operator limitation",
            allow_module_level=True,
        )
    raise

# Use linalg-specific square-matrix shapes: one batched small case covers
# the (*, n, n) interface, and 4x4 through 32x32 covers the small/medium
# matrices targeted by this single-program LU implementation.
SLOGDET_SHAPES = [
    (2, 3, 3),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


class SlogdetBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (A,)


@pytest.mark._linalg_slogdet
def test__linalg_slogdet():
    bench = SlogdetBenchmark(
        op_name="_linalg_slogdet",
        torch_op=torch.ops.aten._linalg_slogdet,
        # _linalg_slogdet generated kernel only supports float32 on CUDA.
        dtypes=[torch.float32],
    )
    bench.set_gems(_linalg_slogdet)
    bench.run()
