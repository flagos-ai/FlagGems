import pytest

import flag_gems

from . import base, consts

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dswiglu")
except ImportError:
    TE_OP = None


@pytest.mark.dswiglu
@pytest.mark.skipif(TE_OP is None, reason="TransformerEngine not installed")
def test_dswiglu():
    bench = base.TexGluBackwardBenchmark(
        op_name="dswiglu",
        torch_op=TE_OP,
        gems_op=flag_gems.dswiglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
