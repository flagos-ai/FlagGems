import pytest
import torch

import flag_gems


@pytest.mark.assert_async
@pytest.mark.parametrize(
    "shape, value, expected_err, match_str",
    [
        ((), 1.0, None, None),
        ((2,), 1.0, RuntimeError, "is ambiguous"),
        ((1,), 1.0, None, None),
    ],
)
def test_assert_async(shape, value, expected_err, match_str):
    msg = "Assertion failed!"
    inp_pt = torch.full(shape, value, device=flag_gems.device)
    inp_triton = inp_pt.clone()
    if expected_err:
        with flag_gems.use_gems():
            with pytest.raises(expected_err, match=match_str):
                flag_gems._assert_async(inp_triton, msg)
                if value == 0:
                    torch.cuda.synchronize()
    else:
        with flag_gems.use_gems():
            flag_gems._assert_async(inp_triton, msg)
            torch.cuda.synchronize()

    if expected_err:
        with pytest.raises(expected_err, match=match_str):
            torch._assert_async(inp_pt, msg)
            if value == 0:
                torch.cuda.synchronize()
    else:
        torch._assert_async(inp_pt, msg)
        torch.cuda.synchronize()
