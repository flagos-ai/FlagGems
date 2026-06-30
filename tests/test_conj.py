import torch
from flag_gems.ops.conj import conj


def test_conj_basic():
    """测试基本共轭功能"""
    x = torch.tensor([1+2j, 3-4j, 5+6j], dtype=torch.complex64)
    y = conj(x)
    y_ref = torch.conj(x)
    assert torch.allclose(y, y_ref), "Conj result mismatch!"
    assert y.data_ptr() != x.data_ptr(), "No physical copy!"
    print("Basic test passed!")


def test_conj_real_input():
    """测试实数输入（应返回副本）"""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = conj(x)
    assert torch.allclose(y, x), "Real input test failed!"
    assert y.data_ptr() != x.data_ptr(), "No physical copy for real input!"
    print("Real input test passed!")


def test_conj_physical_copy():
    """测试物理内存复制是否独立"""
    x = torch.tensor([1+2j, 3-4j], dtype=torch.complex64)
    y = conj(x)
    x[0] = 10+10j
    expected = torch.tensor([1-2j, 3+4j], dtype=torch.complex64)
    assert torch.allclose(y, expected), "Physical copy test failed: output changed!"
    print("Physical copy test passed!")


if __name__ == "__main__":
    test_conj_basic()
    test_conj_real_input()
    test_conj_physical_copy()
    print("All tests passed! ")
