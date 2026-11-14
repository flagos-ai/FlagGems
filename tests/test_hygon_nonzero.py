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

import importlib

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.nonzero,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon override"),
]


def _check(inp, as_tuple):
    ref = torch.nonzero(inp, as_tuple=as_tuple)
    original = inp.to(flag_gems.device)
    result = flag_gems.nonzero(original, as_tuple=as_tuple)
    assert type(result) is type(ref)
    if not as_tuple:
        result, ref = (result,), (ref,)
    assert len(result) == len(ref)
    for actual, expected in zip(result, ref):
        assert actual.dtype == torch.int64
        assert actual.device == original.device
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (2, 0, 3),
        (),
        (1,),
        (1023,),
        (1024,),
        (1025,),
        (2049,),
        (2637,),
        (8193,),
        (33, 31),
        (2, 3, 5, 7),
    ],
)
@pytest.mark.parametrize("density", [0.0, 0.01, 0.5, 1.0])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_hygon_nonzero_cardinality(shape, density, as_tuple):
    generator = torch.Generator(device="cpu").manual_seed(20261104)
    inp = torch.randn(shape, generator=generator)
    inp *= torch.rand(shape, generator=generator) < density
    _check(inp, as_tuple)


@pytest.mark.parametrize("n", [2**22 - 1, 2**22, 2**22 + 1])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_hygon_nonzero_scan_boundary(n, as_tuple):
    inp = torch.zeros(n, dtype=torch.float32)
    inp[0], inp[1023], inp[1024], inp[-1] = 1.0, -1.0, float("nan"), 2.0
    backend = importlib.import_module("flag_gems.runtime.backend._hygon.ops.nonzero")
    device_inp = inp.to(flag_gems.device)
    assert backend._can_use_block_scan(device_inp) == (n <= 2**22)
    _check(inp, as_tuple)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice", "expand"])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_hygon_nonzero_fallback(dtype, layout, as_tuple):
    # Transfer the initialized full base, then construct the view on device.
    base = torch.arange(6 * 35, dtype=torch.int64).remainder(3).to(dtype)
    inp = base.to(flag_gems.device).reshape(6, 35)
    if layout == "transpose":
        inp = inp.t()
    elif layout == "slice":
        inp = inp[1:5, 3:34:2]
    elif layout == "expand":
        inp = inp[1:2].expand(6, 35)
    if layout != "contiguous":
        assert not inp.is_contiguous()
    if layout in ("slice", "expand"):
        assert inp.storage_offset() > 0
    backend = importlib.import_module("flag_gems.runtime.backend._hygon.ops.nonzero")
    assert backend._can_use_block_scan(inp) == (
        dtype == torch.float32 and layout == "contiguous"
    )
    expected = torch.nonzero(inp.cpu(), as_tuple=as_tuple)
    actual = flag_gems.nonzero(inp, as_tuple=as_tuple)
    if not as_tuple:
        actual, expected = (actual,), (expected,)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.dtype == torch.int64 and a.device == inp.device
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


@pytest.mark.parametrize("as_tuple", [False, True])
def test_hygon_nonzero_strided_fp32(as_tuple):
    # Keep offset and noncontiguity on device: Tensor.to may densify CPU views.
    base = torch.arange(6 * 35, dtype=torch.float32, device=flag_gems.device)
    inp = base.reshape(6, 35)[1:5, 3:34:2]
    backend = importlib.import_module("flag_gems.runtime.backend._hygon.ops.nonzero")
    assert not backend._can_use_block_scan(inp)
    assert inp.storage_offset() > 0 and not inp.is_contiguous()
    expected = torch.nonzero(inp.cpu(), as_tuple=as_tuple)
    actual = flag_gems.nonzero(inp, as_tuple=as_tuple)
    if not as_tuple:
        actual, expected = (actual,), (expected,)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1e-30, -1e-30],
        [float("nan")],
        [0.0],
        [-0.0],
    ],
)
def test_hygon_nonzero_special(values, as_tuple):
    inp = torch.tensor(values, dtype=torch.float32)
    if len(values) == 1:
        inp = inp.reshape(())
    _check(inp, as_tuple)


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_hygon_nonzero_scalar_fallback(value, as_tuple):
    _check(torch.tensor(value, dtype=torch.int64), as_tuple)


def test_hygon_nonzero_export_and_stream():
    assert flag_gems.nonzero.__module__.endswith("_hygon.ops.nonzero")
    cpu = torch.tensor([[0.0, float("nan")], [-1.0, 0.0]])
    expected = torch.nonzero(cpu)
    # Hygon's normal runtime device is CUDA/HIP; no Torch-FL install hook.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inp = cpu.to(flag_gems.device)
        # The pytest framework owns dispatch; direct use_gems is forbidden in
        # operator tests because it can bypass reference-as-candidate checks.
        actual = flag_gems.nonzero(inp, as_tuple=False)
        repeated = flag_gems.nonzero(inp, as_tuple=False)
    stream.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated.cpu(), expected, rtol=0, atol=0)
    assert actual.data_ptr() != repeated.data_ptr()
