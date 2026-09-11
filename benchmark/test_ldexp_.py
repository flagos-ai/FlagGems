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


class LdexpInplaceBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1,),
            (1024,),
            (1024, 1024),
            (4096, 4096),
            (16, 128, 64, 64),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(self.set_more_shapes())

    def set_more_shapes(self):
        return [(2**24,), (8192, 4096)]


def ldexp_inplace_input_fn(shape, dtype, device):
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    yield self, other


@pytest.mark.ldexp_
def test_ldexp_():
    bench = LdexpInplaceBenchmark(
        op_name="ldexp_",
        torch_op=torch.ops.aten.ldexp_.default,
        input_fn=ldexp_inplace_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
