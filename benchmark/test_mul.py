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


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES is included (Issue #2693).
@pytest.mark.mul
def test_mul():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
        # dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.mul_
def test_mul_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul_",
        torch_op=lambda a, b: a.mul_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
