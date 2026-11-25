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

    TE_OP = getattr(tex, "dswiglu", None)
except ImportError:
    TE_OP = None
except RuntimeError as te_import_error:
    # The installed transformer-engine here is only the PyPI metapackage, so
    # transformer_engine.common aborts while importing and turns the module
    # uncollectable.  There is no GCU baseline for the TE reference op (torch
    # aten/TE not supported on this backend), so skip on enflame with the real
    # cause instead of failing collection for every operator of the run.
    # enflame has no TE extension, so it measures against eager torch autograd
    # instead (see `test_dswiglu`); the other vendors keep failing loudly
    # because they are expected to ship a TE baseline.
    if flag_gems.vendor_name != "enflame":
        raise
    TE_IMPORT_ERROR = te_import_error
    TE_OP = None


def _te_dswiglu(grad_output, inp, quantizer=None):
    if flag_gems.vendor_name == "kunlunxin":
        return TE_OP(grad_output, inp)
    return TE_OP(grad_output, inp, quantizer)


def enflame_dswiglu_golden(grad_output, inp, quantizer=None):
    """Eager torch dSwiGLU through autograd (silu/mul forward and backward)."""
    x = inp.detach().clone().requires_grad_(True)
    gate, shift = x.chunk(2, dim=-1)
    out = torch.nn.functional.silu(gate) * shift
    (grad_in,) = torch.autograd.grad(out, (x,), (grad_output,))
    return grad_in


@pytest.mark.dswiglu
def test_dswiglu():
    if TE_OP is None and flag_gems.vendor_name != "enflame":
        pytest.skip("'dswiglu' not found in TransformerEngine")
    torch_op = _te_dswiglu if TE_OP is not None else enflame_dswiglu_golden
    bench = base.TexGluBackwardBenchmark(
        op_name="dswiglu",
        torch_op=torch_op,
        gems_op=flag_gems.dswiglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
