import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, INT_DTYPES, gems_assert_equal, to_reference

CAT_SHAPES = [
    [(1, 32), (8, 32)],
    [(16, 128), (32, 128)],
    [(1024, 1024), (1024, 1024)],
    [(1, 1024, 256), (8, 1024, 256), (16, 1024, 256)],
    [(16, 320, 15), (32, 320, 15), (64, 320, 15)],
    [(16, 128, 64, 64), (16, 128, 64, 64), (24, 128, 64, 64), (32, 128, 64, 64)],
]


def gen_cat_shapes_dim(shapes):
    results = []
    for tensor_shapes in shapes:
        assert all(
            [len(s) == len(tensor_shapes[0]) for s in tensor_shapes]
        ), "All tensor rank must agree."
        assert all(
            [s[-1] == tensor_shapes[0][-1] for s in tensor_shapes]
        ), "All tensor must have same shape except cat dim."
        rank = len(tensor_shapes[0])
        results.append([tensor_shapes, 0])
        for dim in range(1, rank):
            results.append(
                [[(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes], dim]
            )
            results.append(
                [
                    [(s[dim], *s[1:dim], s[0], *s[dim + 1 :]) for s in tensor_shapes],
                    dim - rank,
                ]
            )
    return results


@pytest.mark.concatenate
@pytest.mark.parametrize("shape, dim", gen_cat_shapes_dim(CAT_SHAPES))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_concatenate(shape, dim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.concatenate(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.concatenate(inp, dim)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.concatenate
@pytest.mark.parametrize(
    "shape, dim",
    [
        (((0, 3), (2, 3)), 0),
        (((0, 3), (0, 3)), 0),
        (((0,), (0,)), 0),
        (((0,), (1, 3)), -1),
        (((0,), (1, 2, 3)), -2),
        (((0,), (1, 1, 2, 3)), -3),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_concatenate_empty_tensor(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    ref_inp = [to_reference(_) for _ in inp]
    ref_out = torch.concatenate(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.concatenate(inp, dim)
    gems_assert_equal(res_out, ref_out)
