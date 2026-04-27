import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.one_hot
def test_one_hot():
    gems_one_hot = flag_gems.one_hot

    dev_type = torch.device(device).type
    expected_device = "cpu" if cfg.TO_CPU else device

    x = torch.tensor([3, 4, 1, 0], device=device, dtype=torch.int64)
    t = gems_one_hot(x)
    expected = torch.tensor(
        [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
        device=expected_device,
    )
    utils.gems_assert_equal(t, expected)

    t = gems_one_hot(x, -1)
    expected = torch.tensor(
        [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
        device=expected_device,
    )
    utils.gems_assert_equal(t, expected)

    t = gems_one_hot(x, 6)
    expected = torch.tensor(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        device=expected_device,
    )
    utils.gems_assert_equal(t, expected)

    x2 = torch.tensor([[3, 4], [1, 0]], device=device, dtype=torch.int64)
    t = gems_one_hot(x2)
    expected = torch.tensor(
        [[[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], [[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]],
        device=expected_device,
    )
    utils.gems_assert_equal(t, expected)

    x0 = torch.tensor(4, device=device, dtype=torch.int64)
    t = gems_one_hot(x0)
    expected = torch.tensor([0, 0, 0, 0, 1], device=expected_device)
    utils.gems_assert_equal(t, expected)

    x_empty = torch.empty([4, 0], dtype=torch.long, device=device)
    t = gems_one_hot(x_empty, 100)
    expected = torch.empty([4, 0, 100], dtype=torch.long, device=expected_device)
    utils.gems_assert_equal(t, expected)

    if dev_type not in ("cuda", "xla", "mps"):
        bad = torch.tensor([3, 4, -1, 0], dtype=torch.long)
        with pytest.raises(RuntimeError):
            gems_one_hot(bad.to(device), -1)

        bad = torch.tensor([3, 4, 1, 0], dtype=torch.long)
        with pytest.raises(RuntimeError):
            gems_one_hot(bad.to(device), 3)

    with pytest.raises(RuntimeError):
        gems_one_hot(torch.empty([4, 0], dtype=torch.long, device=device))

    with pytest.raises(RuntimeError):
        gems_one_hot(torch.tensor([3, 4, 1, 0], dtype=torch.long, device=device), -2)
