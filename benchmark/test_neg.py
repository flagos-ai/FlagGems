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

import flag_gems

from . import base, consts


@pytest.mark.neg
def test_neg():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg", torch_op=torch.neg, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.neg_
def test_neg_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="neg_", torch_op=torch.neg_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()


class HygonNegBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype):
        # Launch-bound sizes, masked tails and a bandwidth-bound case. The
        # transposed/expanded cases measure generic fallback overhead too.
        for size in (1, 1023, 1024, 1025, 4096, 65536, 1048576):
            yield (torch.randn((size,), dtype=cur_dtype, device=self.device),)
        matrix = torch.randn((128, 256), dtype=cur_dtype, device=self.device)
        yield (matrix.t(),)
        yield (matrix[:, ::2],)
        yield (matrix[:1].expand(128, 256),)


@pytest.mark.neg
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon neg specialization")
def test_neg_hygon():
    bench = HygonNegBenchmark(
        op_name="neg",
        torch_op=torch.neg,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )
    bench.run()
