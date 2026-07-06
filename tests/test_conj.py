import torch

from flag_gems.ops.conj import conj, conj_physical
from flag_gems.testing import assert_close


def test_conj_lazy_view():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=torch.complex64).to(device)

    y = conj(x)
    y_ref = torch.conj(x)

    assert_close(y, y_ref, dtype=torch.complex64)
    assert y.data_ptr() == x.data_ptr(), "Lazy conj should share memory!"
    print("test_conj_lazy_view passed!")


def test_conj_physical_basic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=torch.complex64).to(device)

    y = conj_physical(x)
    y_ref = torch.conj_physical(x)

    assert_close(y, y_ref, dtype=torch.complex64)
    assert y.data_ptr() != x.data_ptr(), "No physical copy!"
    print("test_conj_physical_basic passed!")


def test_conj_physical_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 4, dtype=torch.complex64, device=device)

    y = conj_physical(x)
    assert y.shape == x.shape, "Shape mismatch!"
    print("test_conj_physical_shape passed!")


def test_conj_physical_real_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)

    y = conj_physical(x)
    assert_close(y, x, dtype=torch.float32)
    assert y.data_ptr() != x.data_ptr(), "Real input should return a copy!"
    print("test_conj_physical_real_input passed!")


if __name__ == "__main__":
    test_conj_lazy_view()
    test_conj_physical_basic()
    test_conj_physical_shape()
    test_conj_physical_real_input()
    print("All tests passed!")
