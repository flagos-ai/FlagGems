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


@pytest.mark.rad2deg
def test_rad2deg():
    bench = base.UnaryPointwiseBenchmark(
        op_name="rad2deg",
        torch_op=torch.rad2deg,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rad2deg_
def test_rad2deg_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="rad2deg_",
        torch_op=torch.Tensor.rad2deg_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
