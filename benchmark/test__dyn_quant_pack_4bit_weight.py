import pytest
import torch

from . import base


@pytest.mark._dyn_quant_pack_4bit_weight
def test__dyn_quant_pack_4bit_weight():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_dyn_quant_pack_4bit_weight",
        torch_op=torch._dyn_quant_pack_4bit_weight,
        dtypes=[torch.float32],
    )
    bench.run()
