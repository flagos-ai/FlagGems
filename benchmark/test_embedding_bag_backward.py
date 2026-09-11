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
import triton
from packaging import version

import flag_gems
from flag_gems import _embedding_bag, _embedding_bag_backward

from .test_embedding_bag import EmbeddingBagBenchmark

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "metax")
    and version.parse(triton.__version__) < version.parse("3.5"),
    reason="CANN 8.5 and MACA 3.7.2 are accuracy-only validation targets",
)


BACKWARD_CASES = [
    pytest.param(
        dtype,
        mode,
        weighted,
        frequency,
        sparse,
        marks=[
            pytest.mark.skipif(
                dtype == torch.float64 and not flag_gems.runtime.device.support_fp64,
                reason="Device does not support float64",
            ),
            pytest.mark.skipif(
                dtype == torch.bfloat16 and not flag_gems.runtime.device.support_bf16,
                reason="Device does not support bfloat16",
            ),
            pytest.mark.skipif(
                dtype == torch.bfloat16 and mode == 2 and not sparse,
                reason="Native dense MAX embedding bag backward does not dispatch bfloat16",
            ),
        ],
    )
    for dtype in [torch.float16, torch.float32, torch.bfloat16, torch.float64]
    for mode, weighted, frequency, sparse in [
        (0, False, False, False),
        (1, False, False, False),
        (2, False, False, False),
        (0, True, False, False),
        (0, False, True, False),
        (1, False, True, False),
        (0, False, False, True),
        (1, False, False, True),
        (2, False, False, True),
    ]
]


class EmbeddingBagBackwardBenchmark(EmbeddingBagBenchmark):
    def get_input_iter(self, cur_dtype):
        for forward_args in super().get_input_iter(cur_dtype):
            weight, indices, offsets, _, _, _, psw, _, _ = forward_args
            output, mapping, sizes, maximum = _embedding_bag(*forward_args)
            grad = torch.randn_like(output)
            yield (
                grad,
                indices,
                offsets,
                mapping,
                sizes,
                maximum,
                weight.shape[0],
                self.frequency,
                self.bag_mode,
                self.sparse,
                psw,
                0,
            )


@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Native aten::_embedding_bag_backward falls back to CPU on CANN 8.5 and 9.0",
)
@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize(
    "dtype,mode,weighted,frequency,sparse",
    BACKWARD_CASES,
)
def test_embedding_bag_backward(dtype, mode, weighted, frequency, sparse):
    suffix = f"_mode{mode}"
    suffix += "_weighted" if weighted else ""
    suffix += "_frequency" if frequency else ""
    suffix += "_sparse" if sparse else ""
    bench = EmbeddingBagBackwardBenchmark(
        op_name=f"embedding_bag_backward{suffix}",
        torch_op=torch.ops.aten._embedding_bag_backward.default,
        gems_op=_embedding_bag_backward,
        dtypes=[dtype],
        mode=mode,
        weighted=weighted,
        frequency=frequency,
        sparse=sparse,
        backward=True,
    )
    bench.run()
