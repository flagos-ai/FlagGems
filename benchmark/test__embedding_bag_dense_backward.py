import pytest
import torch

from . import base


@pytest.mark._embedding_bag_dense_backward
def test__embedding_bag_dense_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_embedding_bag_dense_backward",
        torch_op=torch._embedding_bag_dense_backward,
        dtypes=[torch.float32],
    )
    bench.run()
