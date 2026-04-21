import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


@pytest.mark.grouped_mm
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
@pytest.mark.parametrize("groups, N, K", GNK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_mm(groups, N, K, dtype):
    assert dtype == torch.bfloat16
    group_A_list = []
    group_B_list = []
    M_list = []
    A_offs = 0
    B_offs = 0

    for _ in range(groups):
        M_g = random.randint(1, 16384)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device="cuda", dtype=dtype)
        B_g = torch.rand([K_g, N_g], device="cuda", dtype=dtype)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        M_list.append(M_g)
        A_offs += M_g
        B_offs += K_g

    mat_a = torch.cat([x for x in group_A_list], dim=0)
    mat_b = torch.stack([x for x in group_B_list], dim=0)
    offs = torch.tensor(
        [sum(M_list[: i + 1]) for i in range(groups)],
        dtype=torch.int32,
        device="cuda",
    )

    ref_out = torch._grouped_mm(mat_a, mat_b, offs)
    with flag_gems.use_gems():
        res_out = torch._grouped_mm(mat_a, mat_b, offs)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
