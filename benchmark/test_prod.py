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


@pytest.mark.prod
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_prod():
    bench = base.UnaryReductionBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def _prod_dim_int_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": len(shape) - 1}


class ProdDimIntBenchmark(base.GenericBenchmark):
    """torch.prod reduction over the last dimension (prod_dim_int)."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (64, 512, 512),
        ]
        self.shape_desc = "input shape, reduced over the last dim"


@pytest.mark.prod_dim_int
def test_perf_prod_dim_int():
    bench = ProdDimIntBenchmark(
        input_fn=_prod_dim_int_input_fn,
        op_name="prod_dim_int",
        torch_op=torch.prod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
