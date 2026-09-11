import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.scatter_reduce
@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("reduce", ["add", "multiply"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_scatter_reduce_repeated_indices(reduce, inplace, dtype):
    inp = torch.ones((2, 3), dtype=dtype, device=flag_gems.device)
    src = torch.full((4, 3), 2.0, dtype=dtype, device=flag_gems.device)
    index = torch.tensor([[0, 1, 0], [1, 0, 1]] * 2, device=flag_gems.device)
    reference = torch.scatter(
        utils.to_reference(inp),
        0,
        utils.to_reference(index),
        utils.to_reference(src),
        reduce=reduce,
    )
    if inplace:
        result = flag_gems.scatter_(inp, 0, index, src, reduce=reduce)
        assert result is inp
    else:
        result = flag_gems.scatter(inp, 0, index, src, reduce=reduce)
        assert torch.equal(inp, torch.ones_like(inp))
    utils.gems_assert_equal(result, reference)
