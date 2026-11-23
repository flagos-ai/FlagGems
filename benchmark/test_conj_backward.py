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

import flag_gems

from . import base, consts


@pytest.mark.conj_backward
def test_conj_backward():
    # conj_backward only operates on complex dtypes (FLOAT_DTYPES not applicable).
    # The reference is torch.conj_physical, the materializing conjugate: the
    # FlagGems gradient materializes conj(grad_output) into a fresh tensor, so
    # the lazy-view torch.conj (a constant ~3.4us metadata flip) would report a
    # meaningless <1 speedup on every shape.
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj_backward",
        torch_op=torch.conj_physical,
        gems_op=flag_gems.conj_backward,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
