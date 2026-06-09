import pytest

import flag_gems

from . import base, consts

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "reglu")
except ImportError:
    TE_OP = None


@pytest.mark.reglu
@pytest.mark.skipif(TE_OP is None, reason="TransformerEngine not installed")
def test_reglu():
    bench = base.TexGluForwardBenchmark(
        op_name="reglu",
        torch_op=TE_OP,
        gems_op=flag_gems.reglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
