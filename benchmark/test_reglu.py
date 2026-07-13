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


def te_reglu_op(input_tensor, quantizer=None):
    # TransformerEngine's reglu kernel only accepts 2-D inputs on this backend
    # (its cast_gated kernel asserts input.data.shape.size() == 2), so collapse
    # all leading dims before calling it and restore the output shape afterwards.
    shape = input_tensor.shape
    last_dim = shape[-1]
    input_2d = input_tensor.contiguous().view(-1, last_dim)
    out = TE_OP(input_2d, quantizer)
    return out.view(*shape[:-1], last_dim // 2)


@pytest.mark.reglu
@pytest.mark.skipif(TE_OP is None, reason="TransformerEngine not installed")
def test_reglu():
    bench = base.TexGluForwardBenchmark(
        op_name="reglu",
        torch_op=te_reglu_op,
        gems_op=flag_gems.reglu,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
