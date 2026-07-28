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


@pytest.mark.conj
def test_conj():
    # _conj only operates on complex dtypes (FLOAT_DTYPES not applicable)
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj",
        torch_op=torch._conj,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
