import pytest

import flag_gems

from . import base, consts

# Note: Importing transformer_engine (especially in some versions like py 3.10) may automatically
# configure the Root Logger (adding handlers). This may cause subsequent `logging.basicConfig`
# calls (used by FlagGems benchmark) to be ignored/no-op, leading to missing result log files.
# See: https://github.com/NVIDIA/TransformerEngine/issues/1065
try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dgeglu")
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    TE_OP = None


@pytest.mark.dgeglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'dgeglu' not found in TransformerEngine")
def test_dgeglu():
    bench = base.TexGluBackwardBenchmark(
        op_name="dgeglu",
        torch_op=TE_OP,
        gems_op=flag_gems.dgeglu,
        dtypes=consts.FLOAT_DTYPES,
        # TODO(Qiming): Is this flag correct?
        is_backward=False,
    )
    bench.run()
