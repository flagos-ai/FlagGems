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

"""Performance of ``swiglu_group_quant`` against an unfused torch baseline.

The baseline is the same computation in torch: a split, a SwiGLU in fp32, a
block amax, and a cast, each of which reads or writes the whole activation.

The comparison that decides whether this operator is worth dispatching is
against the vendor kernel (``aclnnSwigluGroupQuant``), not against torch; that
table lives in the operator's module docstring, and
``tests/test_swiglu_group_quant.py`` holds the byte-exactness check against it.
"""

import pytest
import torch

import flag_gems

from . import base

# 1/fp8max as the vendor's exact fp32 constant; see the operator module.
INV_FP8_E4M3_MAX = (
    torch.tensor([0x3B124925], dtype=torch.int32).view(torch.float32).item()
)
MX_BLOCK = 32
EPS = 1e-4


def torch_swiglu_group_quant(x):
    """Unfused torch equivalent of ``quant_mode=1`` (MX, block 32, e4m3)."""
    D = x.shape[-1]
    H = D // 2
    xf = x.reshape(-1, D).to(torch.float32)
    x0, x1 = xf[:, :H], xf[:, H:]
    y = (x0 / (1.0 + torch.exp(-x0))) * x1
    yb = y.reshape(-1, H // MX_BLOCK, MX_BLOCK)
    amax = yb.abs().amax(-1).clamp_min(EPS)
    sf = amax * INV_FP8_E4M3_MAX
    bits = sf.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    inv = (((127 - e) & 0xFF) << 23).to(torch.int32).view(torch.float32)
    yq = (yb * inv[:, :, None]).reshape(-1, H).to(torch.float8_e4m3fn)
    return yq, (e + 127).to(torch.uint8)


def _input_fn(shape, dtype, device):
    T, D = shape
    yield (torch.randn(T, D, dtype=dtype, device=device) * 3,)


class SwigluGroupQuantBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPE_DESC = "T, D"

    def set_more_shapes(self):
        # D must be a multiple of 256, so the generic shape sweep does not
        # apply; the shapes in core_shapes.yaml are the MoE sizes that matter.
        return []


@pytest.mark.swiglu_group_quant
@pytest.mark.skipif(
    "swiglu_group_quant" not in vars(flag_gems),
    reason="backend does not export swiglu_group_quant",
)
def test_swiglu_group_quant_perf():
    bench = SwigluGroupQuantBenchmark(
        op_name="swiglu_group_quant",
        input_fn=_input_fn,
        torch_op=torch_swiglu_group_quant,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(flag_gems.swiglu_group_quant)
    bench.run()
