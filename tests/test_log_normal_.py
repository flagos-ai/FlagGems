import pytest
import torch

import flag_gems


@pytest.mark.log_normal_
@pytest.mark.parametrize("shape", [(1024,), (256, 256), (32, 64, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_log_normal_(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref = inp.clone()

    ref.log_normal_()
    with flag_gems.use_gems():
        inp.log_normal_()

    # log-normal values must be positive
    assert (inp > 0).all()
