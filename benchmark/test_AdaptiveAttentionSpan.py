import pytest

import flag_gems

from . import base, consts


@pytest.mark.AdaptiveAttentionSpan
def test_AdaptiveAttentionSpan():
    bench = base.UnaryPointwiseBenchmark(
        op_name="AdaptiveAttentionSpan",
        torch_op=flag_gems.adaptive_attention_span,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
