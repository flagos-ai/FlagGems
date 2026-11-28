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

    TE_OP = getattr(tex, "reglu", None)
except ImportError:
    TE_OP = None
except RuntimeError as te_import_error:
    # The installed transformer-engine here is only the PyPI metapackage, so
    # transformer_engine.common aborts while importing and turns the module
    # uncollectable.  There is no GCU baseline for the TE reference op (torch
    # aten/TE not supported on this backend), so skip on enflame with the real
    # cause instead of failing collection for every operator of the run.
    if flag_gems.vendor_name == "enflame":
        pytest.skip(
            "enflame GCU: transformer-engine metapackage has no framework "
            f"extension ({te_import_error}); the TE reference baseline is not "
            "available on this backend (torch aten/TE not supported on GCU)",
            allow_module_level=True,
        )
    raise


@pytest.mark.reglu
@pytest.mark.skipif(TE_OP is None, reason="'reglu' not found in TransformerEngine")
def test_reglu():
    bench = base.TexGluForwardBenchmark(
        op_name="reglu",
        torch_op=TE_OP,
        gems_op=flag_gems.reglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
