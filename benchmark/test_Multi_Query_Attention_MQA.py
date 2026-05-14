import pytest
import torch

from . import base


@pytest.mark.Multi_Query_Attention_MQA
def test_Multi_Query_Attention_MQA():
    bench = base.UnaryPointwiseBenchmark(
        op_name="Multi_Query_Attention_MQA",
        torch_op=torch.Multi_Query_Attention_MQA,
        dtypes=[torch.float32],
    )
    bench.run()
