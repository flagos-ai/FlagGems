import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_UNSAFE_MASKED_INDEX_PUT_SHAPES = [
    (10,),
    (20,),
    (50,),
    (100,),
]

# tl.atomic_add does not support bfloat16; only float16/float32 are valid
_UNSAFE_MASKED_INDEX_PUT_DTYPES = [torch.float16, torch.float32]


@pytest.mark.unsafe_masked_index_put_accumulate
@pytest.mark.parametrize("shape", _UNSAFE_MASKED_INDEX_PUT_SHAPES)
@pytest.mark.parametrize("dtype", _UNSAFE_MASKED_INDEX_PUT_DTYPES)
def test_unsafe_masked_index_put_accumulate(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    mask = torch.rand(shape) < 0.3
    mask = mask.to(flag_gems.device)

    indices = torch.randint(
        0, max(shape[-1], 1), shape, dtype=torch.long, device=flag_gems.device
    )

    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch._unsafe_masked_index_put_accumulate(
        ref_inp, mask, (indices,), values
    )
    with flag_gems.use_gems():
        res_out = torch._unsafe_masked_index_put_accumulate(
            inp, mask, (indices,), values
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
