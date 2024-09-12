import functools
import operator
from typing import Iterable, Sequence, Tuple

import torch
import triton
import triton.language as tl

Shape = Tuple[int]
Stride = Tuple[int]
MultiIndex = Tuple[int]
Perm = Tuple[int]


def broadcast(s1: Shape, s2: Shape) -> Shape:
    _s1, _s2 = s1, s2
    r1 = len(s1)
    if r1 == 0:
        return s2
    r2 = len(s2)
    if r2 == 0:
        return s1

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    s = list(s1)

    for i in range(r2):
        if s1[d + i] == 1:
            s[d + i] = s2[i]
        elif s2[i] == 1:
            s[d + i] = s1[d + i]
        elif s2[i] == s1[d + i]:
            s[d + i] = s2[i]
        else:
            raise ValueError(f"Unbroadcastable {_s1} and {_s2}")
    s = tuple(s)
    return s


def broadcastable(s1: Shape, s2: Shape) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:
        return True

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    for i in range(r2):
        if s1[d + i] == 1 or s2[i] == 1 or s1[d + i] == s2[i]:
            continue
        return False
    return True


def broadcastable_to(s1: Shape, s2: Shape) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:  # r1 > 0
        return False

    if r1 > r2:
        return False

    d = r2 - r1
    for i in range(r1):
        if s1[i] == 1 or s1[i] == s2[d + i]:
            continue
        return False
    return True


def broadcast_shapes(shapes: Iterable[Shape]) -> Shape:
    if len(shapes) == 0:
        return ()
    shape = shapes[0]
    for s in shapes[1:]:
        shape = broadcast(shape, s)
    return shape


def broadcasted_stride(shape: Shape, stride: Stride, new_shape: Shape) -> Stride:
    assert broadcastable_to(shape, new_shape)
    r1 = len(shape)
    r2 = len(new_shape)
    d = r2 - r1
    new_stride = [0 for _ in range(r2)]
    for i in range(r1):
        new_stride[d + i] = 0 if (shape[i] == 1 and new_shape[d + i] > 1) else stride[i]
    return tuple(new_stride)


def volume(shape: Shape) -> int:
    return functools.reduce(operator.mul, shape, 1)


def is_valid_perm(perm: Perm) -> bool:
    r = len(perm)
    sorted_axes = sorted(perm)
    for i in range(r):
        if sorted_axes[i] != i:
            return False
    return True


def unravel_index(linear_offset: int, shape: Shape) -> MultiIndex:
    multi_index = []
    r = len(shape)
    for i in range(r):
        s = shape[r - 1 - i]
        i = linear_offset % s
        linear_offset = linear_offset // s
        multi_index.append(i)
    return tuple(reversed(multi_index))


def c_contiguous_stride(shape: Shape) -> Stride:
    strides = []
    s = 1
    for size in reversed(shape):
        strides.append(s)
        s *= max(size, 1)  # treat size 0 as size 1
    return tuple(reversed(strides))


def f_contiguous_stride(shape: Shape) -> Stride:
    strides = []
    s = 1
    for size in shape:
        strides.append(s)
        s *= max(size, 1)  # treat size 0 as size 1
    return tuple(strides)


def ordered_stride(shape: Shape, order: Perm) -> Stride:
    strides = [0] * len(shape)
    s = 1
    for i in order:
        strides[i] = s
        s *= max(shape[i], 1)  # treat size 0 as size 1
    return tuple(strides)


def stride_order(strides):
    # we also handle negative strides
    return sorted(range(len(strides)), key=lambda i: abs(strides[i]))


def all_the_same_shape(tensors: Sequence[torch.Tensor]) -> bool:
    if len(tensors) == 0:
        return True
    shape = tensors[0].shape
    return all(item.shape == shape for item in tensors[1:])


def all_the_same_stride(tensors: Sequence[torch.Tensor]) -> bool:
    if len(tensors) == 0:
        return True
    stride = tensors[0].stride()
    return all(item.stride() == stride for item in tensors[1:])


def all_c_contiguous(tensors: Sequence[torch.Tensor]) -> bool:
    if len(tensors) == 0:
        return True
    return all(tensor.is_contiguous() for tensor in tensors)


def heuristics_for_tile_size(max_tile_size, *sizes):
    ndim = len(sizes)
    tile_sizes = [0 for _ in range(ndim)]
    for i in range(ndim):
        size = sizes[ndim - 1 - i]
        tile_size = min(max_tile_size, triton.next_power_of_2(size))
        tile_sizes[ndim - 1 - i] = tile_size
        max_tile_size = max(1, max_tile_size // tile_size)
    return tuple(tile_sizes)


# This should be part of CodeGenConfig
def heuristics_for_num_warps(tile_size):
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def size_in_bytes(a):
    return a.numel() * a.element_size()


def can_use_int32_index(a):
    INT32_MAX = torch.iinfo(torch.int32).max
    if a.is_contiguous():
        return size_in_bytes(a) <= INT32_MAX

    max_offset = 0
    for size, stride in zip(a.shape, a.stride()):
        max_offset += size * stride
        if max_offset > INT32_MAX:
            return False
    return True


def offsetCalculator(inp, idx, strides, dim, isInp):
    ndim = inp.ndim
    shape = list(inp.shape)
    offsets = 0
    idx_dim = 0
    for d in range(0, ndim):
        mod = idx % shape[d]
        add_on = mod * strides[d]
        offsets += add_on
        if d == dim:
            idx_dim = add_on
        idx = idx // shape[d]
        # FIXME: Should we write a fast div/mod
        # to boost the '%' and '//'? (Since they may be run many times)
        # See also:
        #   - https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
        #   - Division by Invariant Integers Using Multiplication,
        #     Torbjörn Granlund and Peter L. Montgomery, 1994.
    return (offsets) if not isInp else (offsets - idx_dim)


def restride_dim(src, dim, shape, step=0, storage_offset=None):
    strides = list(src.stride())
    strides[dim] *= step
    return src.as_strided(shape, strides, storage_offset)


def cfggen():
    block_m = [1, 2, 4]
    block_n = [256, 1024, 2048, 4096]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=4)
        for m in block_m
        for n in block_n
    ]
    return configs


@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def add_on_kernel(
    idx,
    add_on,
    cur_shape,
    cur_strides,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    rows_offset = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offset < M

    cols_offset = pid_y + tl.arange(0, BLOCK_N)[None, :]
    cols_mask = cols_offset < N
    block_mask = rows_mask and cols_mask

    offsets = rows_offset * N + cols_offset
    cur_idx = tl.load(idx + offsets, mask=block_mask, other=1)
    mod = cur_idx % cur_shape
    res = mod * cur_strides
    tl.store(add_on + offsets, res, mask=block_mask)


def offset_calculator(inp, idx, strides, dim, isInp):
    ndim = inp.ndim
    shape = list(inp.shape)
    offsets = torch.zeros_like(inp, dtype=torch.int32, device=inp.device)
    idx_dim = torch.zeros_like(inp, dtype=torch.int32, device=inp.device)
    for d in range(0, ndim):
        add_on = torch.zeros_like(inp, dtype=torch.int32, device=inp.device)
        N = idx.size(idx.ndim - 1)
        M = idx.numel() // N
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        add_on_kernel[grid](idx, add_on, shape[d], strides[d], M, N)

        offsets = torch.add(offsets, add_on)
        if d == dim:
            idx_dim = add_on
        idx = idx // shape[d]
    return offsets if not isInp else (offsets - idx_dim)
