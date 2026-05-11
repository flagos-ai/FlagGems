import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad3d
@pytest.mark.parametrize(
    "shape", [(1, 3, 4, 8, 8), (2, 16, 2, 3, 5), (4, 8, 3, 4, 4), (2, 1, 1, 2, 2)]
)
@pytest.mark.parametrize("padding", [1, (1, 2, 0, 1, 2, 0), 2, (0, 0, 1, 2, 3, 0)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d(shape, padding, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    m_ref = torch.nn.ReplicationPad3d(padding)
    ref = m_ref(x)
    ref_out = utils.to_reference(ref, True)
    with flag_gems.use_gems():
        res_out_functional = flag_gems.replication_pad3d(x, padding)

    utils.gems_assert_close(res_out_functional, ref_out, dtype, reduce_dim=1)
