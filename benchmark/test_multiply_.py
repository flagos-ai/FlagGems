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

from . import base, consts


@pytest.mark.multiply_
def test_multiply_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="multiply_",
        torch_op=lambda a, b: a.multiply_(b),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
