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

"""End-to-end attention-call timing, including quantization and allocation.

H3-shaped local attention probes, not distributed TP/SP or full-model timing.
Head counts 28/14/7 simulate local head shards; communication is not measured.
"""

import pytest
import torch

import flag_gems

from . import base


class SageBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(44, 44, 56)] + [
        (n, n, h) for n in (8598, 34498, 75938) for h in (56, 28, 14, 7)
    ]
    DEFAULT_SHAPE_DESC = "NQ, NK, H (B=1, D=128; local attention only)"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from inputs(shape, dtype, self.device)


def inputs(shape, dtype, device):
    nq, nk, h = shape
    q = torch.randn((1, nq, h, 128), device=device, dtype=dtype)
    k, v = [torch.randn((1, nk, h, 128), device=device, dtype=dtype) for _ in range(2)]
    yield q, k, v


@pytest.mark.sage_attention
def test_sage_attention():
    if flag_gems.vendor_name != "nvidia" or torch.cuda.get_device_capability() != (
        9,
        0,
    ):
        pytest.skip("The same-arithmetic native FP8 Sage benchmark requires SM90")
    native = pytest.importorskip("sageattention").sageattn_qk_int8_pv_fp8_cuda_sm90

    def reference(q, k, v):
        return native(
            q,
            k,
            v,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_thread",
            pv_accum_dtype="fp32+fp32",
        )

    def candidate(q, k, v):
        return flag_gems.sage_attention(q, k, v, pv_dtype="fp8", implementation="sm90")

    SageBenchmark(
        op_name="sage_attention",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.bfloat16],
    ).run()
