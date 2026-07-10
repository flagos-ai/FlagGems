import pytest

import flag_gems

from . import base, consts

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


def te_swiglu_op(input_tensor, quantizer=None):
    # TransformerEngine's swiglu requires a 2-D input on some backends (e.g. musa),
    # while FlagGems supports arbitrary shapes by flattening to 2-D internally.
    # Collapse the leading dims before calling tex.swiglu and restore the output
    # shape so the comparison stays valid across vendors.
    shape = input_tensor.shape
    last_dim = shape[-1]
    ref_input = input_tensor.view(-1, last_dim)
    out = TE_OP(ref_input, quantizer)
    return out.view(*shape[:-1], last_dim // 2)


@pytest.mark.swiglu
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine not installed")
@pytest.mark.skipif(TE_OP is None, reason="'swilu' not found in TransformerEngine")
@pytest.mark.skipif(GEMS_OP is None, reason="'swiglu' not found in FlagGems")
def test_swiglu():
    bench = base.TexGluForwardBenchmark(
        op_name="swiglu",
        torch_op=te_swiglu_op,
        gems_op=GEMS_OP,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
