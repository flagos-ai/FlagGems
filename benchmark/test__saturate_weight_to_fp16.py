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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts

# ``_saturate_weight_to_fp16`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._saturate_weight_to_fp16`` and ``-m
# _saturate_weight_to_fp16`` both work.
setattr(
    pytest.mark,
    "_saturate_weight_to_fp16",
    MarkDecorator(
        Mark("_saturate_weight_to_fp16", (), {}, _ispytest=True), _ispytest=True
    ),
)


@pytest.mark._saturate_weight_to_fp16
def test__saturate_weight_to_fp16():
    # Note: PyTorch's _saturate_weight_to_fp16 has a broken CPU implementation
    # that crashes, so we benchmark against torch.clamp as the reference
    def reference_op(x):
        return torch.clamp(x, -65504.0, 65504.0)

    bench = base.UnaryPointwiseBenchmark(
        op_name="_saturate_weight_to_fp16",
        torch_op=reference_op,
        gems_op=flag_gems._saturate_weight_to_fp16,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
