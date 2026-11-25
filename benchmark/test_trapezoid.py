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

from . import base, consts, utils


def _x_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype=dtype, device=device)
    x = torch.sort(
        utils.generate_tensor_input((shape[-1],), dtype=dtype, device=device)
    )[0]
    yield inp, x


class TrapezoidBenchmark(base.GenericBenchmark):
    # torch's reference trapezoid(y, x) allocates x (same size as y) plus several
    # full-size intermediates, so drop the very largest default shapes that would
    # otherwise OOM, and cap the extra shapes to keep memory bounded.
    _MAX_NUMEL = 2**28

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [s for s in self.shapes if math.prod(s) <= self._MAX_NUMEL]

    def set_more_shapes(self):
        more_shapes_1d = [
            (2**22,),
        ]
        more_shapes_2d = [(10000, 2**i) for i in (0, 8, 12)]
        more_shapes_3d = [(100, 2**i, 100) for i in (0, 8, 12)]
        return more_shapes_1d + more_shapes_2d + more_shapes_3d


@pytest.mark.trapezoid
def test_trapezoid():
    # The trapezoid.x overload (per-sample-point spacing); the trapezoid.dx
    # overload is served by the `trapz` operator and benchmarked there.
    bench = TrapezoidBenchmark(
        op_name="trapezoid",
        torch_op=torch.trapezoid,
        input_fn=_x_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
