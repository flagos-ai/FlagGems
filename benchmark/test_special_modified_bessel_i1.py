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

from . import base

# torch.special.modified_bessel_i1 CUDA reference only supports float32
# (not fp16/bf16), so benchmark dtypes are restricted to float32.
_DTYPES = [torch.float32]


@pytest.mark.special_modified_bessel_i1
def test_special_modified_bessel_i1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_modified_bessel_i1",
        torch_op=torch.special.modified_bessel_i1,
        dtypes=_DTYPES,
    )
    bench.run()


@pytest.mark.special_modified_bessel_i1_out
def test_special_modified_bessel_i1_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_modified_bessel_i1_out",
        torch_op=torch.special.modified_bessel_i1,
        dtypes=_DTYPES,
    )
    bench.run()
