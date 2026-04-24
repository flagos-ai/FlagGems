import pytest
import torch
from flaggems.ops.leaky_relu import leaky_relu, leaky_relu_

SHAPES = [(1,), (64, 64), (1024 * 1024,), (2, 3, 4, 5)]
DTYPES = [torch.float32, torch.float16]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("contiguous", [True, False])
def test_leaky_relu_correctness(shape, dtype, inplace, contiguous):
    torch.manual_seed(42)

    if contiguous:
        x_tri = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    else:
        # 制造步长为 2 的非连续内存
        shape_double = list(shape)
        shape_double[0] *= 2
        x_tri = torch.randn(
            shape_double, dtype=dtype, device="cuda", requires_grad=True
        )[::2]

    x_ref = x_tri.clone().detach().requires_grad_(True)

    # 避免 in-place 操作破坏叶子节点
    x_in_tri = x_tri.clone() if inplace else x_tri
    x_in_ref = x_ref.clone() if inplace else x_ref

    out_ref = torch.nn.functional.leaky_relu(
        x_in_ref, negative_slope=0.01, inplace=inplace
    )
    out_tri = leaky_relu_(x_in_tri, 0.01) if inplace else leaky_relu(x_in_tri, 0.01)

    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    torch.testing.assert_close(out_tri, out_ref, atol=atol, rtol=rtol)


def test_leaky_relu_exceptions():
    with pytest.raises(TypeError):
        leaky_relu(torch.tensor([1, 2], dtype=torch.int32, device="cuda"))
