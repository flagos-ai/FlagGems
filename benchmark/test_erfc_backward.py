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

from flag_gems.ops.erfc_backward import ErfcFunction

from . import base, consts


@pytest.mark.erfc_backward
def test_erfc_backward():
    # ``aten::erfc`` keeps its own autograd formula node (pow/neg/exp/mul), so
    # drive the gradient through the FlagGems autograd bridge explicitly --
    # otherwise this would time the native four-kernel chain, not this kernel.
    bench = base.UnaryPointwiseBenchmark(
        op_name="erfc_backward",
        torch_op=torch.erfc,
        gems_op=ErfcFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
