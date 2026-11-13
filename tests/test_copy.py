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
import math

import pytest
import torch

import flag_gems
from flag_gems.ops.copy import _can_use_triton

from . import accuracy_utils as utils


def _is_float8_dtype_supported(dtype: torch.dtype) -> bool:
    if dtype is None:
        return False
    if str(flag_gems.device).startswith("cuda"):
        device_index = 0
        if isinstance(flag_gems.device, str) and ":" in flag_gems.device:
            device_index = int(flag_gems.device.split(":")[1])

        cap = torch.cuda.get_device_capability(device_index)
        if cap[0] < 8 or (cap[0] == 8 and cap[1] < 9):
            return False
    try:
        t = torch.zeros(1, device=flag_gems.device, dtype=dtype)
        return t.dtype == dtype
    except (RuntimeError, TypeError):
        return False


_FLOAT8_DTYPES = []
for _dtype_name in ["float8_e4m3fn", "float8_e5m2"]:
    _dtype = getattr(torch, _dtype_name, None)
    if _dtype is not None and _is_float8_dtype_supported(_dtype):
        _FLOAT8_DTYPES.append(_dtype)


@pytest.mark.copy_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    (
        utils.FLOAT_DTYPES + [torch.int32, torch.int64, torch.int8, torch.uint8]
        if flag_gems.vendor_name == "cambricon"
        else utils.FLOAT_DTYPES + [torch.int8, torch.uint8]
    ),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_same_dtype(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        if dtype in utils.FLOAT_DTYPES:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        else:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
    else:
        if dtype in [torch.int8, torch.uint8]:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
        else:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_dst = torch.zeros_like(ref_src)
    res_dst = torch.zeros_like(src)

    ref_dst.copy_(ref_src)
    flag_gems.copy_(res_dst, src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_broadcast():
    dst_shape = (2, 3)
    src = torch.arange(0, 3, dtype=torch.float32, device=flag_gems.device)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(
        torch.zeros(dst_shape, dtype=torch.float32, device=flag_gems.device)
    )
    res_dst = torch.zeros(dst_shape, dtype=torch.float32, device=flag_gems.device)

    ref_dst.copy_(ref_src)
    flag_gems.copy_(res_dst, src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_dtype_fallback():
    src = torch.arange(0, 8, dtype=torch.int32, device=flag_gems.device)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(
        torch.zeros(src.shape, dtype=torch.float32, device=flag_gems.device)
    )
    res_dst = torch.zeros(src.shape, dtype=torch.float32, device=flag_gems.device)

    ref_dst.copy_(ref_src)
    flag_gems.copy_(res_dst, src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="PyTorch does not support float8_e8m0fnu",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="mthreads does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="MetaX does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend does not support float8_e8m0 dtypes",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="KUNLUNXIN does not support float8_e8m0fnu dtype",
)
@pytest.mark.parametrize(
    "shape",
    [
        (8,),
        (4, 4),
        (2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
        (100, 200),
        (90, 60, 30),
        (1000, 2001),
        (7, 13, 17),
        (3, 31, 61, 127),
    ],
)
def test_copy_inplace_float8_e8m0fnu(shape):
    """Test that copy_ works correctly with float8_e8m0fnu (e8m0) dtype tensors.

    Triton does not recognize float8_e8m0fnu, so FlagGems should fallback to
    PyTorch's native copy_ implementation for this dtype.
    """
    device = flag_gems.device

    # e8m0 is an exponent-only format, create via view from uint8
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support uint8 generation.
        src_uint8 = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            device
        )
    else:
        src_uint8 = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
    src = src_uint8.view(torch.float8_e8m0fnu)
    ref_src = utils.to_reference(src)

    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support float8_e8m0fnu generation.
        ref_dst = utils.to_reference(
            torch.zeros(shape, dtype=torch.float8_e8m0fnu, device="cpu").to(device)
        )
        res_dst = torch.zeros(shape, dtype=torch.float8_e8m0fnu, device="cpu").to(
            device
        )
    else:
        ref_dst = utils.to_reference(
            torch.zeros(shape, dtype=torch.float8_e8m0fnu, device=device)
        )
        res_dst = torch.zeros(shape, dtype=torch.float8_e8m0fnu, device=device)
    ref_dst.copy_(ref_src)

    flag_gems.copy_(res_dst, src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="PyTorch does not support float8_e8m0fnu",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="mthreads does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="MetaX does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="KUNLUNXIN does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend NPU does not support float8_e8m0fnu dtype for copy_d2d",
)
def test_copy_inplace_float8_e8m0fnu_to_float32():
    """Test copy_ from float8_e8m0fnu to float32."""
    device = flag_gems.device
    shape = (8,)

    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support uint8 generation.
        src_uint8 = torch.randint(1, 200, shape, dtype=torch.uint8, device="cpu").to(
            device
        )
    else:
        src_uint8 = torch.randint(1, 200, shape, dtype=torch.uint8, device=device)
    src = src_uint8.view(torch.float8_e8m0fnu)
    ref_src = utils.to_reference(src)

    ref_dst = utils.to_reference(torch.zeros(shape, dtype=torch.float32, device=device))
    res_dst = torch.zeros(shape, dtype=torch.float32, device=device)
    ref_dst.copy_(ref_src)

    flag_gems.copy_(res_dst, src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.int32),
        (torch.int16, torch.float32),
        (torch.bool, torch.float32),
        (torch.int8, torch.float32),
        (torch.uint8, torch.float16),
        (torch.bool, torch.int8),
        (torch.bool, torch.uint8),
        (torch.float32, torch.bool),
        (torch.float16, torch.uint8),
        (torch.int8, torch.bool),
        (torch.uint8, torch.bool),
    ],
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_mixed_dtype_triton(src_dtype, dst_dtype):
    device = flag_gems.device
    numel = 8

    if src_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        if src_dtype == torch.uint8:
            src = torch.randint(0, numel, (numel,), dtype=src_dtype, device=device)
        else:
            src = torch.randint(-numel, numel, (numel,), dtype=src_dtype, device=device)
    elif src_dtype is torch.bool:
        base = torch.tensor([True, False, True, True, False, True, False, True])
        src = base.to(device=device)
    else:
        if flag_gems.vendor_name == "mthreads":
            src = torch.arange(numel, device="cpu", dtype=src_dtype).to(device)
        else:
            src = torch.arange(numel, device=device, dtype=src_dtype)

    dst = torch.zeros(numel, dtype=dst_dtype, device=device)

    assert _can_use_triton(dst, src)

    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())
    ref_dst.copy_(ref_src)

    res_dst = dst.clone()
    flag_gems.copy_(res_dst, src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    (
        utils.FLOAT_DTYPES + [torch.int32, torch.int64, torch.int8, torch.uint8]
        if flag_gems.vendor_name == "cambricon"
        else utils.FLOAT_DTYPES + [torch.int8, torch.uint8]
    ),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_functional_same_dtype(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        if dtype in utils.FLOAT_DTYPES:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        else:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
    else:
        if dtype in [torch.int8, torch.uint8]:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
        else:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    template = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_template = utils.to_reference(template)

    ref_out = torch.ops.aten.copy(ref_template, ref_src)
    res_out = flag_gems.copy(template, src)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.copy
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_functional_broadcast():
    src = torch.arange(0, 3, dtype=torch.float32, device=flag_gems.device)
    template = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_template = utils.to_reference(template)

    ref_out = torch.ops.aten.copy(ref_template, ref_src)
    res_out = flag_gems.copy(template, src)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.copy
@pytest.mark.skipif(
    len(_FLOAT8_DTYPES) == 0, reason="No float8 is supported in current environment"
)
@pytest.mark.parametrize("dtype", _FLOAT8_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (8,),
        (4, 4),
        (2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ],
)
def test_copy_functional_float8(dtype, shape):
    device = flag_gems.device

    src_uint8 = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)

    src = src_uint8.view(dtype)
    ref_src = utils.to_reference(src)

    template = torch.zeros(shape, dtype=dtype, device=device)

    ref_dst = torch.empty_like(ref_src)

    ref_dst.copy_(ref_src)
    res_dst = flag_gems.copy(template, src)

    utils.gems_assert_equal(res_dst.view(torch.uint8), ref_dst.view(torch.uint8))


# Ascend coreDim and large-tensor regressions.
CORE_DIM_LIMIT = 65535
# Keep the default destination aligned for int32 reinterpretation. Unaligned
# source/destination offsets are exercised separately.
GUARD = 40
DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
]
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.fixture
def release_device_memory():
    # Only the real large-input cases need allocator cleanup between cases.
    torch.npu.empty_cache()
    yield
    torch.npu.empty_cache()


@pytest.fixture
def copy_ops():
    ops = importlib.import_module(flag_gems.copy_.__module__)
    assert ops.__name__ == flag_gems.copy.__module__
    # Backend import errors can silently select the generic implementation.
    assert "_ascend" in ops.__name__
    return ops


@pytest.fixture
def checked_launches(copy_ops, monkeypatch):
    """Reject invalid grids before launching; record the effective tile inputs."""
    kernel = copy_ops._flat_copy_kernel
    launches = []

    class CheckedKernel:
        def __getitem__(self, grid):
            resolved = grid(None) if callable(grid) else grid
            assert 0 < math.prod(resolved) <= CORE_DIM_LIMIT

            def checked(*args, **kwargs):
                launches.append((resolved, args[2], args[3], args[4], args[5]))
                return kernel[resolved](*args, **kwargs)

            return checked

    monkeypatch.setattr(copy_ops, "_flat_copy_kernel", CheckedKernel())
    return launches


def _bytes(tensor):
    # Materialize reference layouts on the CPU, independently of NPU kernels.
    return tensor.cpu().contiguous().reshape(-1).view(torch.uint8)


def _assert_same_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(_bytes(actual), _bytes(expected), rtol=0, atol=0)


def _filled(shape, dtype, value, device=None):
    """A tensor whose every byte is ``value``, including noncanonical bool bytes."""
    numel = math.prod(shape)
    esize = torch.empty((), dtype=dtype).element_size()
    flat = torch.full((numel * esize,), value, dtype=torch.uint8, device=device)
    return flat.view(dtype).reshape(shape)


def _payload(shape, dtype, seed):
    """Deterministic bytes without changing the process-wide random state."""
    numel = math.prod(shape)
    esize = torch.empty((), dtype=dtype).element_size()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, (numel,), dtype=dtype, generator=generator).reshape(
            shape
        )
    raw = torch.randint(
        0, 256, (numel * esize,), dtype=torch.uint8, generator=generator
    )
    # Every source byte differs from the destination sentinel.
    raw[raw == 0x5A] = 0x5B
    return raw.view(dtype).reshape(shape)


def _check_copy(payload_cpu, src_view=lambda x: x, dst_offset=GUARD):
    """Check values, in-place identity and both destination storage guards."""
    expected = src_view(payload_cpu)
    src = src_view(payload_cpu.to(flag_gems.device))
    numel = src.numel()
    base = _filled((dst_offset + numel + GUARD,), src.dtype, 0x5A, flag_gems.device)
    dst = base[dst_offset : dst_offset + numel].reshape(src.shape)
    pointer, offset, strides = dst.data_ptr(), dst.storage_offset(), dst.stride()

    result = flag_gems.copy_(dst, src)
    torch.npu.synchronize()

    assert result is dst
    assert result.data_ptr() == pointer
    assert result.storage_offset() == offset
    assert result.stride() == strides
    _assert_same_bytes(result, expected)
    _assert_same_bytes(src, expected)
    esize = base.element_size()
    raw = _bytes(base)
    guards = torch.cat((raw[: dst_offset * esize], raw[-GUARD * esize :]))
    torch.testing.assert_close(guards, torch.full_like(guards, 0x5A), rtol=0, atol=0)


def _check_functional_copy(template, src, expected):
    template_before = _bytes(template).clone()
    src_before = _bytes(src).clone()
    out = flag_gems.copy(template, src)
    torch.npu.synchronize()
    assert out.shape == template.shape
    assert out.dtype == template.dtype
    assert out.device == template.device
    assert out.stride() == template.stride()
    # data_ptr() is zero for empty tensors, so compare storage identity instead.
    assert not torch._C._is_alias_of(out, template)
    assert not torch._C._is_alias_of(out, src)
    _assert_same_bytes(out, expected)
    torch.testing.assert_close(_bytes(template), template_before, rtol=0, atol=0)
    torch.testing.assert_close(_bytes(src), src_before, rtol=0, atol=0)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,shape,n_elements",
    [(dtype, (17,), 34 if dtype == torch.int64 else 17) for dtype in DTYPES]
    + [
        (dtype, (words * 4 // torch.empty((), dtype=dtype).element_size(),), words)
        for dtype in (torch.int8, torch.bfloat16, torch.float32)
        for words in (4095, 4096, 4097)
    ]
    + [(torch.bool, (), 1), (torch.bfloat16, (), 1), (torch.int64, (), 2)]
    + [(torch.float32, (3, 5, 1371), 20565)],
)
def test_copy_contiguous_boundaries_exact(checked_launches, dtype, shape, n_elements):
    _check_copy(_payload(shape, dtype, 1))
    assert len(checked_launches) == 1
    grid, actual_n, block, masked, wide = checked_launches[0]
    # Check boundaries in the kernel's units, after any int32 reinterpretation.
    assert actual_n == n_elements
    assert grid == ((n_elements + block - 1) // block,)
    assert masked == (n_elements % block != 0)
    assert not wide


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,dst_shape,src_shape,valid",
    [
        (torch.bool, (0,), (0,), True),
        (torch.float32, (2, 0, 3), (2, 0, 3), True),
        (torch.int8, (2, 0, 3), (1, 1, 3), True),
        (torch.float32, (0,), (2,), False),
        (torch.float32, (2, 0, 3), (2, 0, 4), False),
    ],
)
def test_copy_empty_exact(dtype, dst_shape, src_shape, valid):
    ref_src = torch.zeros(src_shape, dtype=dtype)
    ref_dst = torch.empty(dst_shape, dtype=dtype)
    src = ref_src.to(flag_gems.device)
    dst = torch.empty(dst_shape, dtype=dtype, device=flag_gems.device)
    if not valid:
        with pytest.raises(RuntimeError):
            ref_dst.copy_(ref_src)
        with pytest.raises(RuntimeError):
            flag_gems.copy_(dst, src)
        return
    ref_dst.copy_(ref_src)
    result = flag_gems.copy_(dst, src)
    assert result is dst
    _assert_same_bytes(result, ref_dst)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,layout",
    [
        (torch.int8, layout)
        for layout in ("transpose", "permute", "step", "slice", "diagonal")
    ]
    + [
        (torch.bool, "stride2"),
        (torch.uint8, "stride7"),
        (torch.int8, "stride3"),
        (torch.int16, "slice"),
        (torch.float16, "diagonal"),
        (torch.bfloat16, "transpose"),
        (torch.float32, "stride3"),
    ],
)
def test_copy_noncontiguous_source_exact(monkeypatch, dtype, layout):
    def view(x):
        if layout.startswith("stride"):
            return x[:: int(layout[len("stride") :])]
        tensor = x.reshape(5, 7, 11)
        if layout == "transpose":
            return tensor.transpose(0, 2)
        if layout == "permute":
            return tensor.permute(1, 2, 0)
        if layout == "step":
            return tensor[1::2, :, 1::3]
        if layout == "slice":
            return tensor[:, :, 2:7]
        return tensor.diagonal(dim1=1, dim2=2)

    payload = _payload(
        (4096,) if layout.startswith("stride") else (5 * 7 * 11,), dtype, 3
    )
    assert not view(payload).is_contiguous()
    if dtype == torch.int8 and layout == "stride3":
        contiguous = torch.Tensor.contiguous

        def reject_device_materialization(tensor, *args, **kwargs):
            if tensor.device.type != "cpu" and not tensor.is_contiguous():
                raise AssertionError(
                    "copy_ must not call contiguous() on its strided source"
                )
            return contiguous(tensor, *args, **kwargs)

        # contiguous() can re-enter copy_ under dispatch. Check the prohibited
        # operation without requiring a particular private kernel name.
        monkeypatch.setattr(torch.Tensor, "contiguous", reject_device_materialization)
    _check_copy(payload, src_view=view)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,layout",
    [
        (torch.bool, "step"),
        (torch.int8, "step"),
        (torch.uint8, "transpose"),
        (torch.float32, "transpose"),
    ],
)
def test_copy_noncontiguous_destination_exact(dtype, layout):
    numel = 2048
    payload = _payload((numel,), dtype, 5)
    storage_size = numel * 2 if layout == "step" else numel
    base = _filled((storage_size + 2 * GUARD,), dtype, 0x5A, flag_gems.device)
    expected_base = _filled(base.shape, dtype, 0x5A)

    def view(x):
        body = x[GUARD:-GUARD]
        return body[::2] if layout == "step" else body.reshape(32, 64).t()

    dst = view(base)
    expected_dst = view(expected_base)
    payload = payload.reshape(dst.shape)
    expected_dst.copy_(payload)
    src = payload.to(flag_gems.device)
    metadata = (dst.data_ptr(), dst.storage_offset(), dst.stride())
    assert not dst.is_contiguous()
    result = flag_gems.copy_(dst, src)
    torch.npu.synchronize()
    assert result is dst
    assert (dst.data_ptr(), dst.storage_offset(), dst.stride()) == metadata
    _assert_same_bytes(result, payload)
    # Include holes between selected elements and guards on both ends.
    _assert_same_bytes(base, expected_base)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "src_dtype,dst_dtype,strided",
    [
        (torch.bool, torch.float32, True),
        (torch.uint8, torch.int32, True),
        (torch.int8, torch.float32, True),
        (torch.int16, torch.int32, True),
        (torch.float16, torch.float32, True),
    ]
    + [
        (src_dtype, dst_dtype, False)
        for src_dtype in (torch.float32, torch.float16, torch.bfloat16)
        for dst_dtype in (torch.uint8, torch.int8, torch.int16)
    ]
    + [
        (src_dtype, torch.bool, False)
        for src_dtype in (torch.float32, torch.int32, torch.int8)
    ],
)
def test_copy_dtype_conversion_matches_cpu(src_dtype, dst_dtype, strided):
    if strided:
        if src_dtype.is_floating_point:
            payload = (torch.arange(4096, dtype=torch.float32) - 2048).to(src_dtype)
        else:
            payload = _payload((4096,), src_dtype, 12)
        src = payload.to(flag_gems.device)[::2]
        reference = payload[::2]
        assert not src.is_contiguous()
    elif dst_dtype == torch.bool:
        reference = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=src_dtype)
        src = reference.to(flag_gems.device)
    else:
        info = torch.iinfo(dst_dtype)
        values = [-1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5]
        values += [info.min, info.min + 1, info.max - 1, info.max, info.max // 2]
        rounded = torch.tensor(values, dtype=src_dtype)
        # Check after rounding, in float64: fp16/bf16 can round an in-range
        # boundary out of range. Out-of-range float casts vary by platform.
        exact = rounded.to(torch.float64)
        reference = rounded[(exact >= info.min) & (exact <= info.max)]
        src = reference.to(flag_gems.device)

    expected = torch.empty(reference.shape, dtype=dst_dtype)
    expected.copy_(reference)
    dst = _filled(expected.shape, dst_dtype, 0x5A, flag_gems.device)
    result = flag_gems.copy_(dst, src)
    assert result is dst
    _assert_same_bytes(result, expected)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,strided",
    [
        (torch.bool, False),
        (torch.int8, False),
        (torch.bfloat16, False),
        (torch.bool, True),
        (torch.int8, True),
        (torch.float16, True),
    ],
)
def test_copy_broadcast_sources_exact(dtype, strided):
    full = _payload((2, 256) if strided else (1, 128), dtype, 13)
    device_full = full.to(flag_gems.device)
    src = device_full[:1, ::2] if strided else device_full
    reference = full[:1, ::2] if strided else full
    assert src.is_contiguous() != strided
    expected = reference.expand(64, 128)
    dst = _filled((64, 128), dtype, 0x5A, flag_gems.device)
    result = flag_gems.copy_(dst, src)
    assert result is dst
    _assert_same_bytes(result, expected)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "src_offset,dst_offset", [(1, 0), (2, 0), (3, 0), (0, 1), (1, 3)]
)
def test_copy_unaligned_storage_offset_exact(checked_launches, src_offset, dst_offset):
    payload = _payload((1024 + src_offset,), torch.int8, 7)
    _check_copy(
        payload, src_view=lambda x: x[src_offset:], dst_offset=GUARD + dst_offset
    )
    # Even if only one operand is unaligned, neither may be viewed as int32.
    assert len(checked_launches) == 1
    assert checked_launches[0][1] == 1024


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_copy_preserves_special_values_exact(dtype):
    info = torch.finfo(dtype)
    values = [
        float("nan"),
        -float("nan"),
        float("inf"),
        -float("inf"),
        0.0,
        -0.0,
        info.max,
        -info.max,
        info.tiny,
        -info.tiny,
        info.eps,
    ]
    payload = torch.tensor(values, dtype=dtype).repeat(1000)[:8191]
    _check_copy(payload)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.float32])
def test_copy_across_devices_exact(dtype):
    payload = _payload((1024,), dtype, 8)
    dst = _filled((1024,), dtype, 0x5A, flag_gems.device)
    assert flag_gems.copy_(dst, payload) is dst
    _assert_same_bytes(dst, payload)
    back = _filled((1024,), dtype, 0x5A)
    assert flag_gems.copy_(back, dst) is back
    _assert_same_bytes(back, payload)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
def test_copy_across_devices_is_refused_when_re_entered(copy_ops, monkeypatch):
    monkeypatch.setattr(copy_ops._reentry, "transferring", True, raising=False)
    host = torch.arange(16, dtype=torch.float32)
    dst = torch.zeros(16, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(NotImplementedError, match="between devices"):
        copy_ops.copy_(dst, host)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
def test_copy_identical_alias_is_a_no_op():
    base = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    expected = torch.arange(16, dtype=torch.float32)
    result = flag_gems.copy_(base, base)
    assert result is base
    _assert_same_bytes(base, expected)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,case",
    [
        (torch.float32, "contiguous"),
        (torch.bfloat16, "tail"),
        (torch.int8, "strided"),
        (torch.bool, "broadcast"),
        (torch.float32, "empty"),
        (torch.bool, "empty_rank3"),
    ],
)
def test_copy_functional_shapes_exact(dtype, case):
    shape = {
        "contiguous": (32,),
        "tail": (4097,),
        "strided": (4096,),
        "broadcast": (1, 128),
        "empty": (0,),
        "empty_rank3": (2, 0, 3),
    }[case]
    payload = _payload(shape, dtype, 15)
    src = payload.to(flag_gems.device)
    if case == "strided":
        src = src[::2]
        payload = payload[::2]
    expected = payload.expand(64, 128) if case == "broadcast" else payload
    template = _filled(expected.shape, dtype, 0x5A, flag_gems.device)
    _check_functional_copy(template, src, expected)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,layout",
    [
        (torch.bool, "step"),
        (torch.int8, "transpose"),
        (torch.float16, "step"),
        (torch.float32, "transpose"),
    ],
)
def test_copy_functional_strided_template_exact(dtype, layout):
    if layout == "transpose":
        base = _filled((32, 64), dtype, 0x5A, flag_gems.device)
        template = base.t()
    else:
        base = _filled((4096,), dtype, 0x5A, flag_gems.device)
        template = base[::2]
    before = _bytes(base).clone()
    payload = _payload(template.shape, dtype, 16)
    assert not template.is_contiguous()
    _check_functional_copy(template, payload.to(flag_gems.device), payload)
    # Check the entire template storage, including holes in a strided view.
    torch.testing.assert_close(_bytes(base), before, rtol=0, atol=0)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
def test_copy_functional_cross_dtype_exact(dtype):
    values = torch.arange(256, dtype=torch.float32) - 128
    template = _filled(values.shape, dtype, 0x5A, flag_gems.device)
    _check_functional_copy(template, values.to(flag_gems.device), values.to(dtype))


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,layout",
    [
        (torch.int8, "cross_dtype"),
        (torch.bfloat16, "cross_dtype"),
        (torch.float32, "step_destination"),
        (torch.float32, "broadcast"),
    ],
)
def test_copy_splits_when_wider_than_pointwise_can_index(
    copy_ops, monkeypatch, dtype, layout
):
    limit = 100
    monkeypatch.setattr(copy_ops, "_MAX_POINTWISE_ELEMS", limit)
    leaves = []
    generic_copy = copy_ops._generic_copy_

    def checked_generic(dst, src, non_blocking=False):
        assert 0 < dst.numel() <= limit
        leaves.append(dst.numel())
        return generic_copy(dst, src, non_blocking)

    monkeypatch.setattr(copy_ops, "_generic_copy_", checked_generic)
    shape = (37, 11)
    payload = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    # Keep all casts finite and in range, independent of backend cast semantics.
    payload = ((payload % 101) - 50).to(dtype)
    if layout == "broadcast":
        payload = payload[:1]
    src = payload.to(flag_gems.device)
    storage_shape = (37, 22) if layout == "step_destination" else shape
    base = _filled(storage_shape, torch.float32, 0x5A, flag_gems.device)
    expected_base = _filled(storage_shape, torch.float32, 0x5A)
    dst = base[:, ::2] if layout == "step_destination" else base
    expected = expected_base[:, ::2] if layout == "step_destination" else expected_base
    expected.copy_(payload)
    result = flag_gems.copy_(dst, src)
    assert result is dst
    _assert_same_bytes(base, expected_base)
    # A correct final value alone would also pass if the split were bypassed.
    assert len(leaves) > 1
    assert sum(leaves) == dst.numel()


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "n_elements",
    [
        1,
        4095,
        4096,
        4097,
        65535 * 4096 - 1,
        65535 * 4096,
        65535 * 4096 + 1,
        65536 * 4096 + 17,
        2**31,
    ],
)
def test_flat_grid_never_exceeds_core_dim(copy_ops, n_elements):
    grid = copy_ops._flat_grid(n_elements)
    tiles = (n_elements + copy_ops._FLAT_COPY_BLOCK - 1) // copy_ops._FLAT_COPY_BLOCK
    assert 0 < math.prod(grid) <= CORE_DIM_LIMIT
    assert math.prod(grid) == min(tiles, CORE_DIM_LIMIT)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,cap",
    [(torch.bool, 1), (torch.int8, 2), (torch.bfloat16, 3), (torch.int64, 2)],
)
def test_copy_capped_grid_covers_every_tile(
    copy_ops, checked_launches, monkeypatch, dtype, cap
):
    monkeypatch.setattr(copy_ops, "_MAX_CORE_DIM", cap)
    numel = 9 * copy_ops._FLAT_COPY_BLOCK + 257
    _check_copy(_payload((numel,), dtype, 10))
    assert checked_launches
    assert all(math.prod(grid) == cap for grid, *_ in checked_launches)


@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "dtype,extra", [(torch.float32, 0), (torch.int8, 7), (torch.bfloat16, 7)]
)
def test_flat_kernel_offset_widths_agree(copy_ops, dtype, extra):
    # Exercise both compiled offset branches on modest inputs. This does not
    # establish correctness at actual offsets beyond int32's range.
    numel = 3 * copy_ops._FLAT_COPY_BLOCK + extra
    expected = _payload((numel,), dtype, 11)
    payload = expected.to(flag_gems.device)
    for wide in (False, True):
        base = _filled((numel + 2 * GUARD,), dtype, 0x5A, flag_gems.device)
        dst = base[GUARD:-GUARD]
        as_int32, n = copy_ops._flat_plan(dst, payload)
        view, out = payload.reshape(-1), dst.reshape(-1)
        if as_int32:
            view, out = view.view(torch.int32), out.view(torch.int32)
        copy_ops._flat_copy_kernel[copy_ops._flat_grid(n)](
            view,
            out,
            n,
            copy_ops._FLAT_COPY_BLOCK,
            n % copy_ops._FLAT_COPY_BLOCK != 0,
            wide,
        )
        torch.npu.synchronize()
        expected_base = _filled(base.shape, dtype, 0x5A)
        expected_base[GUARD:-GUARD].copy_(expected)
        _assert_same_bytes(base, expected_base)


@pytest.mark.usefixtures("release_device_memory")
@pytest.mark.copy
@pytest.mark.copy_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend copy kernel")
@pytest.mark.parametrize(
    "numel,dtype",
    [
        (65535 * 4096 * 2, torch.bfloat16),
        (65536 * 4096 * 2, torch.bfloat16),
        (65536 * 4096 + 17, torch.int32),
    ],
)
def test_copy_real_large_input_exact(checked_launches, numel, dtype):
    pattern = torch.arange(1024, dtype=dtype, device=flag_gems.device)
    src = pattern.repeat(numel // 1024 + 1)[:numel]
    dst = _filled((numel,), dtype, 0x5A, flag_gems.device)
    result = flag_gems.copy_(dst, src)
    torch.npu.synchronize()
    assert result is dst
    for start in range(0, numel, 2**24):
        chunk = slice(start, start + 2**24)
        _assert_same_bytes(result[chunk], src[chunk])
    assert checked_launches
