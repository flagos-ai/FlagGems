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

from . import base, consts, utils

try:
    import torch_npu

    NPU_SWIGLU = getattr(torch_npu, "npu_swiglu", None)
except ImportError:
    torch_npu = None
    NPU_SWIGLU = None

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "swiglu")
    TE_AVAILABLE = True
    GEMS_OP = getattr(flag_gems, "swiglu")
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None
    GEMS_OP = None


@pytest.mark.swiglu
def test_swiglu():
    if TE_AVAILABLE and TE_OP is not None and GEMS_OP is not None:
        bench = base.TexGluForwardBenchmark(
            op_name="swiglu",
            torch_op=TE_OP,
            gems_op=GEMS_OP,
            dtypes=consts.FLOAT_DTYPES,
        )
    elif NPU_SWIGLU is not None:
        # Ascend: TransformerEngine is unavailable, so the golden is torch_npu.
        bench = SwigluBenchmark(
            op_name="swiglu",
            input_fn=utils.unary_input_fn,
            torch_op=npu_swiglu_golden,
            gems_op=flag_gems.swiglu,
            dtypes=consts.FLOAT_DTYPES,
        )
    else:
        pytest.skip(
            "neither TransformerEngine 'swiglu' nor torch_npu.npu_swiglu is available"
        )
    bench.run()


# swiglu splits the last dim in half, so it must be even; shapes come from
# core_shapes.yaml like other ops.
def npu_swiglu_golden(input_tensor: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_swiglu(input_tensor, dim=-1)


class SwigluBenchmark(base.GenericBenchmark):
    """Filters out shapes with an odd last dim, which swiglu cannot accept."""

    def set_more_shapes(self):
        return [s for s in super().set_more_shapes() if s[-1] % 2 == 0]
