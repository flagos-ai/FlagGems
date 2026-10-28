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

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    try:
        from transformer_engine.pytorch import DType as TEDType
    except ImportError:
        from transformer_engine.pytorch.cpp_extensions import DType as TEDType

    TE_OP = getattr(tex, "geglu", None)
    TE_AVAILABLE = True
    GEMS_OP = getattr(flag_gems, "geglu", None)
    TORCH_TO_TE_DTYPE = {
        torch.float32: TEDType.kFloat32,
        torch.float16: TEDType.kFloat16,
        torch.bfloat16: TEDType.kBFloat16,
    }
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None
    GEMS_OP = None
    TORCH_TO_TE_DTYPE = {}


def te_geglu(inp, quantizer=None):
    if base.vendor_name == "kunlunxin":
        return TE_OP(inp, None, None, TORCH_TO_TE_DTYPE[inp.dtype])
    return TE_OP(inp, None)


@pytest.mark.geglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'geglu' not found in TransformerEngine")
@pytest.mark.skipif(GEMS_OP is None, reason="'geglu' not found in FlagGems")
def test_geglu():
    bench = base.TexGluForwardBenchmark(
        op_name="geglu",
        torch_op=te_geglu,
        gems_op=GEMS_OP,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
