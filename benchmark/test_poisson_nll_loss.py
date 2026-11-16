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

from . import base, consts


def _input_fn(shape, dtype, device):
    input = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randint(0, 5, shape, device=device).to(dtype)
    yield input, target


def _torch_poisson_nll_loss(input, target):
    return torch.ops.aten.poisson_nll_loss(input, target, True, False, 1e-8, 1)


class PoissonNllLossBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
            (8192, 8192),
        ]


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss():
    bench = PoissonNllLossBenchmark(
        input_fn=_input_fn,
        op_name="poisson_nll_loss",
        torch_op=_torch_poisson_nll_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
