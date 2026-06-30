import torch
from flag_gems.ops.conj import conj


def test_conj_basic():
    """测试基本共轭功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1+2j, 3-4j, 5+6j], dtype=torch.complex64).to(device)

    y = conj(x)
    y_ref = torch.conj(x)

    print("x:", x)
    print("y (our conj):", y)
    print("y_ref (torch.conj):", y_ref)
    print("diff:", y - y_ref)

    assert torch.allclose(y, y_ref), "Conj result mismatch!"
    assert y.data_ptr() != x.data_ptr(), "No physical copy!"
    print("Basic test passed!")


def test_conj_real_input():
    """测试实数输入（应返回副本）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)

    y = conj(x)
    assert torch.allclose(y, x), "Real input test failed!"
    assert y.data_ptr() != x.data_ptr(), "No physical copy for real input!"
    print("Real input test passed!")


def test_conj_physical_copy():
    """测试物理内存复制是否独立"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1+2j, 3-4j], dtype=torch.complex64).to(device)

    y = conj(x)
    x[0] = 10+10j
    expected = torch.tensor([1-2j, 3+4j], dtype=torch.complex64).to(device)

    assert torch.allclose(y, expected), "Physical copy test failed: output changed!"
    print("Physical copy test passed!")


if __name__ == "__main__":
    test_conj_basic()
    test_conj_real_input()
    test_conj_physical_copy()
    print("All tests passed! 🎉")
