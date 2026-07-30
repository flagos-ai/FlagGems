# Copyright 2026 FlagOS Contributors

import pytest
import torch

from flag_gems.experimental_ops.__rshift__ import (rshift_scalar,
                                                   rshift_scalar_,
                                                   rshift_tensor,
                                                   rshift_tensor_)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64, torch.uint8])
@pytest.mark.parametrize("shape", [(1024,), (7, 13), (2, 3, 5)])
def test_rshift_tensor_and_inplace(dtype, shape):
    value = torch.randint(0, 100, shape, dtype=dtype, device="cuda")
    shift = torch.randint(0, 7, shape, dtype=dtype, device="cuda")

    torch.testing.assert_close(rshift_tensor(value, shift), value >> shift)

    actual = value.clone()
    rshift_tensor_(actual, shift)
    torch.testing.assert_close(actual, value >> shift)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64, torch.uint8])
def test_rshift_scalar_and_inplace(dtype):
    value = torch.randint(0, 100, (11, 17), dtype=dtype, device="cuda")

    torch.testing.assert_close(rshift_scalar(value, 3), value >> 3)

    actual = value.clone()
    rshift_scalar_(actual, 3)
    torch.testing.assert_close(actual, value >> 3)
