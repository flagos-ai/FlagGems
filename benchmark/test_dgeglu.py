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

import inspect

import pytest

import flag_gems

from . import base, consts

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType

    TE_OP = getattr(tex, "dgeglu", None)
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None
    TE_DType = None


def _te_dgeglu_takes_otype():
    """Whether this TransformerEngine build takes the output dtype as the third
    argument of ``dgeglu``.

    Upstream TE takes a quantizer in that slot and accepts ``None`` there, which
    is what the benchmark passes (matching ``flag_gems.dgeglu``'s signature);
    some builds expose the output dtype instead.  Probe once at import time
    rather than per call, so the reference timing stays clean.
    """
    try:
        return "otype" in inspect.signature(TE_OP).parameters
    except (TypeError, ValueError):
        # A compiled binding: the quantizer slot.
        return False


_TE_DGEGLU_OTYPE = TE_OP is not None and _te_dgeglu_takes_otype()


def _te_dgeglu(grad_output, inp, quantizer=None):
    # TransformerEngine's dgeglu takes an extra argument that the benchmark
    # carries for signature compatibility; feed it whichever form this build
    # expects.
    if _TE_DGEGLU_OTYPE:
        return TE_OP(grad_output, inp, TE_DType[inp.dtype])
    return TE_OP(grad_output, inp, quantizer)


@pytest.mark.dgeglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'dgeglu' not found in TransformerEngine")
def test_dgeglu():
    bench = base.TexGluBackwardBenchmark(
        op_name="dgeglu",
        torch_op=_te_dgeglu,
        gems_op=flag_gems.dgeglu,
        dtypes=consts.FLOAT_DTYPES,
        # TODO(Qiming): Is this flag correct?
        is_backward=False,
    )
    bench.run()
