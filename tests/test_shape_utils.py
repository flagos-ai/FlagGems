import torch

from flag_gems.utils import shape_utils


def test_c_contiguous_stride_normal():
    shape = (2, 3, 4)
    assert shape_utils.c_contiguous_stride(shape) == (12, 4, 1)


def test_c_contiguous_stride_with_zero_size():
    shape = (2, 0, 4)
    assert shape_utils.c_contiguous_stride(shape) == (4, 4, 1)


def test_f_contiguous_stride_normal():
    shape = (2, 3, 4)
    assert shape_utils.f_contiguous_stride(shape) == (1, 2, 6)


def test_f_contiguous_stride_with_zero_size():
    shape = (2, 0, 4)
    assert shape_utils.f_contiguous_stride(shape) == (1, 2, 2)


def test_ordered_stride_normal():
    shape = (2, 3, 4)
    stride_order = (0, 2, 1)
    ref_stride = (1, 8, 2)
    assert shape_utils.ordered_stride(shape, stride_order) == ref_stride


def test_ordered_stride_with_zero_size():
    shape = (2, 3, 0)
    stride_order = (0, 2, 1)
    ref_stride = (1, 2, 2)
    assert shape_utils.ordered_stride(shape, stride_order) == ref_stride


def test_stride_order():
    strides = (8, 16, 1)
    assert shape_utils.stride_order(strides) == [2, 0, 1]


def test_all_the_same_shape_empty():
    assert shape_utils.all_the_same_shape([])


def test_all_the_same_shape1():
    xs = [torch.randn(2, 3) for _ in range(3)]
    assert shape_utils.all_the_same_shape(xs)


def test_all_the_same_shape2():
    xs = [torch.randn(2, 3) for _ in range(3)] + [
        torch.randn(
            10,
        )
    ]
    assert shape_utils.all_the_same_shape(xs) is False


def test_all_the_same_stride_empty():
    assert shape_utils.all_the_same_stride([])


def test_all_the_same_stride1():
    xs = [torch.randn(2, 3) for _ in range(3)]
    assert shape_utils.all_the_same_stride(xs)


def test_all_the_same_stride2():
    xs = [torch.randn(2, 3) for _ in range(3)] + [
        torch.randn(
            10,
        )
    ]
    assert shape_utils.all_the_same_stride(xs) is False


def test_all_c_contiguous_empty():
    assert shape_utils.all_c_contiguous([])


def test_all_c_contiguous1():
    xs = [torch.randn(3, 4), torch.randn(2, 3)]
    assert shape_utils.all_c_contiguous(xs)


def test_heuristics_for_tile_size():
    shape = (10000, 10000, 10)
    tile_sizes = (1, 256, 16)
    assert shape_utils.heuristics_for_tile_size(4096, *shape) == tile_sizes


def test_heuristics_for_num_warps():
    assert shape_utils.heuristics_for_num_warps(1024) == 4
    assert shape_utils.heuristics_for_num_warps(2048) == 8
    assert shape_utils.heuristics_for_num_warps(4096) == 16
