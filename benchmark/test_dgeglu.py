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


def _te_dgeglu(grad_output, inp, quantizer=None):
    # TransformerEngine's dgeglu takes an extra argument that the benchmark
    # carries for signature compatibility.  Upstream TE takes a quantizer there
    # and accepts ``None`` (which is what the benchmark passes); the version
    # vendored for kunlunxin takes the output dtype instead.
    if flag_gems.vendor_name == "kunlunxin":
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
