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


@pytest.mark.floor
def test_floor():
    bench = base.UnaryPointwiseBenchmark(
        op_name="floor", torch_op=torch.floor, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.floor_
def test_floor_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="floor_",
        torch_op=torch.Tensor.floor_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.floor_out
def test_floor_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="floor_out",
        torch_op=torch.floor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class FloorContiguousBenchmark(base.UnaryPointwiseBenchmark):
    # Run in comprehensive mode to include both complete tiles and masked tails.
    def set_more_shapes(self):
        return super().set_more_shapes() + [
            (1,),
            (1023,),
            (1024,),
            (1025,),
            (65536,),
            (65537,),
            (1048576,),
            (1048577,),
        ]


@pytest.mark.floor
def test_floor_contiguous_launch():
    bench = FloorContiguousBenchmark(
        op_name="floor", torch_op=torch.floor, dtypes=[torch.float32]
    )
    bench.run()
