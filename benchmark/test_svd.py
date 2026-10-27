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

pytestmark = pytest.mark.filterwarnings(
    "ignore:Warning only once for all operators.*:UserWarning"
)


class SvdBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "(*B), M, N"
    DEFAULT_DTYPES = [torch.float32]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            yield inp, {"some": True, "compute_uv": True}


@pytest.mark.svd
def test_svd():
    bench = SvdBenchmark(op_name="svd", torch_op=torch.svd)
    bench.run()
