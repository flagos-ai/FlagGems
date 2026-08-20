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

    TE_OP = getattr(tex, "dswiglu", None)
except ImportError:
    TE_OP = None


@pytest.mark.dswiglu
@pytest.mark.skipif(TE_OP is None, reason="'dswiglu' not found in TransformerEngine")
def test_dswiglu():
    bench = base.TexGluBackwardBenchmark(
        op_name="dswiglu",
        torch_op=TE_OP,
        gems_op=flag_gems.dswiglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
