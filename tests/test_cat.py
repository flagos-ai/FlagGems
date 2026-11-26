# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    CAT_SHAPES = [
        [(1, 32), (8, 32)],
        [(16, 320, 15), (32, 320, 15), (64, 320, 15)],
    ]
else:
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


@pytest.mark.cat
@pytest.mark.parametrize("shape, dim", gen_cat_shapes_dim(CAT_SHAPES))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_cat(shape, dim, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]
    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    res_out = flag_gems.cat(inp, dim)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cat
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_cat_empty_tensor(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch.cat(ref_inp, dim)

    res_out = flag_gems.cat(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cat
@pytest.mark.parametrize("with_empty", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.skipif(
    flag_gems.cat.__module__ != "flag_gems.ops.cat",
    reason="Generic cat implementation only",
)
def test_cat_single_input_does_not_alias(with_empty, noncontiguous):
    inp = torch.arange(12, device=flag_gems.device, dtype=torch.float32).reshape(3, 4)
    if noncontiguous:
        inp = inp.t()
    inputs = [inp.new_empty(0), inp] if with_empty else [inp]
    expected = torch.cat([utils.to_reference(tensor) for tensor in inputs])

    with flag_gems.use_gems(include=["cat"]):
        result = torch.cat(inputs)

    utils.gems_assert_equal(result, expected)
    result.add_(1)
    utils.gems_assert_equal(inp, expected)


@pytest.mark.cat
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or not hasattr(torch, "float8_e8m0fnu"),
    reason="Requires NVIDIA float8_e8m0fnu",
)
def test_cat_single_float8_input_does_not_alias():
    data = torch.arange(1, 13, device=flag_gems.device, dtype=torch.uint8)
    expected = utils.to_reference(data).clone()
    inp = data.view(torch.float8_e8m0fnu)

    with flag_gems.use_gems(include=["cat"]):
        result = torch.cat([inp])

    assert result.dtype == inp.dtype
    utils.gems_assert_equal(result.view(torch.uint8), expected)
    result.view(torch.uint8).zero_()
    utils.gems_assert_equal(data, expected)


@pytest.mark.cat
@pytest.mark.parametrize("metadata", ["negative", "zero"])
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="Requires NVIDIA tensor metadata support",
)
def test_cat_single_input_with_metadata_does_not_alias(metadata):
    if metadata == "negative":
        inp = torch._neg_view(
            torch.arange(1, 5, device=flag_gems.device, dtype=torch.float32)
        )
    else:
        inp = torch._efficientzerotensor(
            (4,), device=flag_gems.device, dtype=torch.float32
        )
    expected = torch.cat([utils.to_reference(inp)])

    result = flag_gems.cat([inp])

    utils.gems_assert_equal(result, expected)
    assert not torch._C._is_alias_of(result, inp)


@pytest.mark.cat_out
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_cat_out_matches_reference(dtype):
    a = torch.randn(3, 5, dtype=dtype, device=flag_gems.device)
    b = torch.randn(7, 5, dtype=dtype, device=flag_gems.device)
    dim = 0

    ref_a = utils.to_reference(a, True)
    ref_b = utils.to_reference(b, True)
    ref_out = torch.empty((10, 5), dtype=dtype, device=ref_a.device)
    torch.ops.aten.cat.out([ref_a, ref_b], dim, out=ref_out)

    out = torch.empty((10, 5), dtype=dtype, device=flag_gems.device)
    flag_gems.cat_out([a, b], dim, out=out)

    utils.gems_assert_close(out, ref_out, dtype)
