import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    '''
    MNK_SHAPES = [
        (1, 1, 32),
        # tl.load
        (1, 1, 1),
        # tl.load + tma_device
        (1, 2, 2),
        # tma_host: mm.py ForceTmaDevice => tma_device, ForceLoad => tl.load
        (1, 8, 8),
        # 1.a GNDAttention::conv1d, [M, K=4, N=2048]
        (1, 2048, 4),
        (2, 2048, 4),
        (4, 2048, 4),
        (8, 2048, 4),
        (24, 2048, 4),
        (192, 2048, 4),
        (2048, 2048, 4),
        (7168, 2048, 4),
        (16384, 2048, 4),
        # 3.c MoE::shared_expert/down_proj, [M, K=128, N=2048]
        (1, 2048, 128),
        (2, 2048, 128),
        (4, 2048, 128),
        (8, 2048, 128),
        (24, 2048, 128),
        (192, 2048, 128),
        (2048, 2048, 128),
        (7168, 2048, 128),
        (16384, 2048, 128),
        # 1.d GNDAttention::out_proj, 2.b FullAttention::o_proj, [M, K=1024, N=2048]
        (1, 2048, 1024),
        (2, 2048, 1024),
        (4, 2048, 1024),
        (8, 2048, 1024),
        (24, 2048, 1024),
        (192, 2048, 1024),
        (2048, 2048, 1024),
        (7168, 2048, 1024),
        (16384, 2048, 1024),
        # 3.d MoE::shared_expert_gate, [M, K=2048, N=1] Fix gemv
        (1, 1, 2048),
        (2, 1, 2048),
        (4, 1, 2048),
        (8, 1, 2048),
        (24, 1, 2048),
        (192, 1, 2048),
        (2048, 1, 2048),
        (7168, 1, 2048),
        (16384, 1, 2048),
        # 1.c GNDAttention::in_proj_ba, [M, K=2048, N=16]
        (1, 16, 2048),
        (2, 16, 2048),
        (4, 16, 2048),
        (8, 16, 2048),
        (24, 16, 2048),
        (192, 16, 2048),
        (2048, 16, 2048),
        (7168, 16, 2048),
        (16384, 16, 2048),
        # 3.a MoE::gate, [M, K=2048, N=512]
        (1, 512, 2048),
        (2, 512, 2048),
        (4, 512, 2048),
        (8, 512, 2048),
        (24, 512, 2048),
        (192, 512, 2048),
        (2048, 512, 2048),
        (7168, 512, 2048),
        (16384, 512, 2048),
        # 2.a FullAttention::qkv_proj, 3.b MoE::shared_expert/gate_up_proj, [M, K=2048, N=2560]
        (1, 2560, 2048),
        (2, 2560, 2048),
        (4, 2560, 2048),
        (8, 2560, 2048),
        (24, 2560, 2048),
        (192, 2560, 2048),
        (2048, 2560, 2048),
        (7168, 2560, 2048),
        (16384, 2560, 2048),
        # 1.b GNDAttention::in_proj_qkvz, [M, K=2048, N=3072]
        (1, 3072, 2048),
        (2, 3072, 2048),
        (4, 3072, 2048),
        (8, 3072, 2048),
        (24, 3072, 2048),
        (192, 3072, 2048),
        (2048, 3072, 2048),
        (7168, 3072, 2048),
        (16384, 3072, 2048),
        # 4. LMHead [M, K=2048, N=37984]
        (1, 37984, 2048),
        (2, 37984, 2048),
        (4, 37984, 2048),
        (8, 37984, 2048),
        (24, 37984, 2048),
        (192, 37984, 2048),
        (2048, 37984, 2048),
        (7168, 37984, 2048),
        (16384, 37984, 2048),
    ]
    '''
    MNK_SHAPES = (
        [
            # tl.load
            #(1, 1, 1),
            # tl.load + tma_device ForceTmaHost => tma_host
            #(1, 2, 2),
            # tma_host ForceTmaDevice => tma_device, ForceLoad => tl.load
            #(1, 8, 8),
            #(1, 2048, 128),
            # 1.a GNDAttention::conv1d, [M, K=4, N=2048]
            (1, 2048, 4),
            (7168, 2048, 4),
            # 3.c MoE::shared_expert/down_proj, [M, K=128, N=2048]
            (1, 2048, 128),
        ]
    )
    #'''
    FLOAT_DTYPES = [torch.bfloat16]
    #FLOAT_DTYPES = [torch.float32]
    #FLOAT_DTYPES = [torch.bfloat16, torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

MK_SHAPES = (
    [(1, 32)]
    if QUICK_MODE
    else [
        (1, 32),
        (7, 33),
        (31, 65),
        (160, 1024),
        (257, 96),
        (1023, 255),
        (5333, 71),
    ]
)


# TODO: failed at (1, 1, 2)
@pytest.mark.mm1
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
#@pytest.mark.parametrize("b_column_major", [True])
@pytest.mark.parametrize("b_column_major", [False])
#@pytest.mark.parametrize("b_column_major", [True, False])
def test_mm(M, N, K, dtype, b_column_major):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 mm test on tsingmicro platform")

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    if b_column_major:
        mat2 = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    else:
        mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 mm self-transpose test on tsingmicro platform")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    ref_mat = utils.to_reference(mat, True)

    ref_out = torch.mm(ref_mat, ref_mat.t())
    with flag_gems.use_gems():
        res_out = torch.mm(mat, mat.t())

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M, K", MK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mm_out_self_transpose(M, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skipping fp32 mm.out self-transpose test on tsingmicro platform")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    mat = torch.randn((K, M), dtype=dtype, device=flag_gems.device).t()
    out = torch.empty((M, M), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_out = utils.to_reference(out, True)

    torch.mm(ref_mat, ref_mat.t(), out=ref_out)
    with flag_gems.use_gems():
        torch.mm(mat, mat.t(), out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)
