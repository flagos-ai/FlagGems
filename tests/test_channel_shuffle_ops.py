import pytest
import torch
import flag_gems
from tests.accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

def ref_channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    return torch.nn.ChannelShuffle(groups)(x)

@pytest.mark.channel_shuffle
@pytest.mark.parametrize(
    "shape, groups",
    [
        ((2, 8, 3, 3), 2),
        ((1, 12, 5, 6), 3),
        ((4, 16, 32, 32), 4),
        ((2, 10, 7), 5),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_channel_shuffle_accuracy(shape, groups, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_ref = to_reference(x, False)
    y_ref = ref_channel_shuffle(x_ref, groups)
    with flag_gems.use_gems():
        y = flag_gems.channel_shuffle(x, groups)
    gems_assert_close(y, y_ref, dtype)

@pytest.mark.channel_shuffle
def test_channel_shuffle_groups_eq_1():
    x = torch.randn((2, 8, 3, 3), dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        y = flag_gems.channel_shuffle(x, 1)
    assert torch.allclose(y, x)

@pytest.mark.channel_shuffle
def test_channel_shuffle_invalid_groups():
    x = torch.randn((2, 10, 3, 3), dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        with pytest.raises(ValueError):
            flag_gems.channel_shuffle(x, 4)