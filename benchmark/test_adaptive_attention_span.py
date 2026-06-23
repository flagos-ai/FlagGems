import pytest

import flag_gems

from . import base, consts


@pytest.mark.adaptive_attention_span
def test_adaptive_attention_span():
    if flag_gems.vendor_name == "metax":
        pytest.skip("Metax backend CI validates correctness; skip backend benchmark.")

    bench = base.UnaryPointwiseBenchmark(
        op_name="adaptive_attention_span",
        torch_op=flag_gems.adaptive_attention_span,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
