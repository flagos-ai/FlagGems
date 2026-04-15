# tests/test_grid_sample.py
import pytest
import torch
import flag_gems

# 第 1 步冒烟测试：确保接线正常
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sample_smoke(dtype, align_corners):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, C, IH, IW = 1, 3, 8, 8
    OH, OW = 4, 4

    # 1. 制造一些简单的测试数据
    input = torch.randn(N, C, IH, IW, dtype=dtype, device=device)
    # 为了测试 padding，人为制造部分超 [-1, 1] 越界的点 
    grid = (torch.rand(N, OH, OW, 2, dtype=dtype, device=device) * 3) - 1.5 

    # 2. 调用原生 PyTorch 拿到标准答案
    ref_out = torch.nn.functional.grid_sample(input, grid, align_corners=align_corners)

    # 3. 在 FlagGems 接管下运行
    with flag_gems.use_gems():
        our_out = torch.nn.functional.grid_sample(input, grid, align_corners=align_corners)
        
    # 4. 判断结果是否一致
    torch.testing.assert_close(our_out, ref_out)
    print("\n烟雾测试通过！！接管成功并且没有精度损失。")
