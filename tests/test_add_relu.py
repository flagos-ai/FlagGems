import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.underscore_add_relu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_relu(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    # _add_relu computes relu(a + b) = max(0, a + b)
    # Since torch._add_relu is not available on CUDA, use relu(add(...)) as reference
    ref_out = torch.relu(ref_inp1 + ref_inp2)
    inputs = (inp1, inp2)
    before = [
        (x.device, x.dtype, tuple(x.shape), x.stride(), x.data_ptr()) for x in inputs
    ]
    saved = [x.clone() for x in inputs]
    res_out = flag_gems._add_relu(inp1, inp2)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == before[0][0]
    assert res_out.dtype == before[0][1]
    assert tuple(res_out.shape) == before[0][2]
    assert res_out.data_ptr() not in {before[0][4], before[1][4]}
    for value, original, abi in zip(inputs, saved, before):
        assert (value.device, value.dtype, tuple(value.shape)) == abi[:3]
        assert value.stride() == abi[3]
        assert value.data_ptr() == abi[4]
        assert torch.equal(
            value.detach().cpu().reshape(-1).view(torch.uint8),
            original.detach().cpu().reshape(-1).view(torch.uint8),
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
