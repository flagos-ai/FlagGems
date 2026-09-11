import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# This operator operates on int4 values stored in int32 tensors (a subset of utils.INT_DTYPES).
# Reference implementation of int4 packing for comparison
def _reference_int4pack(inp, innerKTiles):
    """Reference implementation of int4 weight packing."""
    M, N = inp.shape
    out = torch.empty((M, N // 2), dtype=torch.uint8, device=inp.device)
    for m in range(M):
        for n in range(N // 2):
            # Pack two int4 values into one byte
            # Lower 4 bits: input[m, 2*n]
            # Upper 4 bits: input[m, 2*n+1]
            low = inp[m, 2 * n].item() & 0xF
            high = inp[m, 2 * n + 1].item() & 0xF
            out[m, n] = (high << 4) | low
    return out


@pytest.mark.convert_weight_to_int4pack
@pytest.mark.parametrize("shape", [(16, 64), (16, 128), (32, 128)])
@pytest.mark.parametrize("innerKTiles", [2, 4, 8])
def test_convert_weight_to_int4pack(shape, innerKTiles):
    """Test _convert_weight_to_int4pack accuracy."""
    M, N = shape
    # Ensure N is divisible by 16 for the operation to work
    # (the operation requires specific alignment)
    if N % 16 != 0:
        N = (N // 16 + 1) * 16
        shape = (M, N)

    # Input values should be in range 0-15 for int4
    # dtype=torch.int32 is required since the input represents raw int4 weight values
    inp = torch.randint(0, 16, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    # Call the gems implementation directly
    with flag_gems.use_gems():
        res_out = flag_gems._convert_weight_to_int4pack(inp, innerKTiles)

    # Validate against a hand-written Python reference implementing the same
    # packing: int32 input -> byte-pair-packed uint8 of shape (M, N//2).
    ref_out = _reference_int4pack(ref_inp, innerKTiles)
    utils.gems_assert_equal(res_out, ref_out)
