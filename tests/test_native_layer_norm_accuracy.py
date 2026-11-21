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

pytestmark = [
    pytest.mark.native_layer_norm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "hygon", reason="Hygon moments regression"
    ),
]


def _check_fields(actual, expected):
    for name, got, ref in zip(("output", "mean", "rstd"), actual, expected):
        got = got.cpu()
        assert got.shape == ref.shape, name
        assert got.dtype == ref.dtype, name
        assert torch.equal(torch.isnan(got), torch.isnan(ref)), name
        assert torch.equal(torch.isinf(got), torch.isinf(ref)), name
        torch.testing.assert_close(
            got,
            ref,
            rtol=1e-4,
            atol=1e-5,
            equal_nan=True,
            msg=lambda msg: name + ": " + msg,
        )


def _case(n, pattern, affine, seed=317, layout="transpose"):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((3, 7, n), generator=generator)
    if pattern == "near_constant":
        x = x * 1e-4 + 3.0
    elif pattern == "large_offset":
        x = x * 1e-2 + 1e4
    elif pattern == "nonfinite":
        x[0, 0, 0] = float("nan")
        x[0, 1, 0] = float("inf")
        x[0, 2, 0] = -float("inf")
        x[1, 0, :] = float("inf")
    if layout == "transpose":
        x = x.transpose(0, 1)
    elif layout == "slice":
        storage = torch.empty((*x.shape[:-1], 2 * n + 1))
        storage[..., 1::2] = x
        x = storage[..., 1::2]
    weight = torch.randn(n, generator=generator) if affine else None
    bias = torch.randn(n, generator=generator) if affine else None
    reference = torch.ops.aten.native_layer_norm.default(x, [n], weight, bias, 1e-5)
    device = flag_gems.device
    dx = x.to(device)
    saved = dx.cpu().clone()
    actual = flag_gems.native_layer_norm(
        dx,
        [n],
        weight.to(device) if weight is not None else None,
        bias.to(device) if bias is not None else None,
        1e-5,
    )
    _check_fields(actual, reference)
    torch.testing.assert_close(dx.cpu(), saved, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("n", [320, 1280])
@pytest.mark.parametrize("pattern", ["near_constant", "nonfinite"])
@pytest.mark.parametrize("affine", [False, True])
def test_native_layer_norm_issue7247(n, pattern, affine):
    _case(n, pattern, affine)


@pytest.mark.parametrize(
    "n", [1, 7, 8, 15, 31, 32, 63, 64, 127, 128, 129, 255, 511, 1024, 4097, 8192, 8193]
)
@pytest.mark.parametrize("seed", [0, 317])
@pytest.mark.parametrize("pattern", ["near_constant", "nonfinite"])
def test_native_layer_norm_moment_boundaries(n, seed, pattern):
    _case(n, pattern, True, seed)


@pytest.mark.parametrize("layout", ["contiguous", "slice", "transpose"])
@pytest.mark.parametrize("pattern", ["near_constant", "large_offset", "random"])
@pytest.mark.parametrize("affine", [False, True])
def test_native_layer_norm_layouts(layout, pattern, affine):
    _case(320, pattern, affine, layout=layout)


@pytest.mark.parametrize("shape", [[], [7], [3, 7, 320, 1]])
def test_native_layer_norm_invalid_shape(shape):
    x = torch.empty((3, 7, 320), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.native_layer_norm(x, shape)


@pytest.mark.parametrize("argument", ["weight", "bias"])
def test_native_layer_norm_invalid_affine(argument):
    x = torch.empty((3, 7, 320), device=flag_gems.device)
    kwargs = {argument: torch.empty(319, device=flag_gems.device)}
    with pytest.raises(RuntimeError):
        flag_gems.native_layer_norm(x, [320], **kwargs)
