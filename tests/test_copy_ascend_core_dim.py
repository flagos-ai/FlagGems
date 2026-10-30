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

"""Ascend copy_ registration, exact bytes, storage safety and launch limits.

Run with the Ascend environment loaded from the repository root::

    python -m pytest tests/test_copy_ascend_core_dim.py --ref=cpu -q

No known failure is skipped or xfailed.

Every comparison is on raw bytes, because a copy performs no arithmetic: any
bit that differs is wrong, and a byte view also catches a launch that silently
dropped its tail. Expected values are written out rather than produced by
``copy_``; importing flag_gems takes over ``aten::copy_`` for CPU tensors as
well, so a ``.copy_()`` reference would be the implementation under test.

The largest case uses about 2 GiB of NPU storage and compares every element in
bounded chunks. The launch-limit cases are planned on the host and allocate
nothing.
"""

import importlib
import math

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.copy,
    pytest.mark.copy_,
    pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend copy kernel"),
]

CORE_DIM_LIMIT = 65535
GUARD = 37
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
ONE_BYTE_DTYPES = [torch.bool, torch.uint8, torch.int8]
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
NARROW_INT_DTYPES = [torch.uint8, torch.int8, torch.int16]
APIS = ["direct"]


@pytest.fixture(autouse=True)
def release_device_memory():
    # Several cases hold a few GiB each. Without this, whether a later one fits
    # depends on what the earlier ones left behind, which is a property of the
    # run rather than of the operator: the functional 2**29 case passed on one
    # machine and hit OutOfMemory on another purely from allocator state.
    yield
    torch.npu.empty_cache()


@pytest.fixture
def copy_ops():
    ops = importlib.import_module(flag_gems.copy_.__module__)
    assert ops.__name__ == flag_gems.copy.__module__
    # A failed import inside _ascend.ops is swallowed by the registrar and every
    # Ascend operator falls back silently, so pin the module down first.
    assert "_ascend" in ops.__name__
    return ops


@pytest.fixture
def checked_launches(copy_ops, monkeypatch):
    """Record every flat launch and reject an invalid grid before it is issued.

    A grid the driver refuses puts the whole process into an error state, after
    which later results in the same process cannot be trusted. Asserting on the
    host keeps a regression readable instead of aborting the run.
    """
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
    return tensor.contiguous().reshape(-1).view(torch.uint8).cpu()


def _assert_same_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(_bytes(actual), _bytes(expected), rtol=0, atol=0)


def _run_copy(dst, src, api):
    # Always the direct call. Registering copy_ for dispatch would need
    # use_gems(), which tools/ci_checks/check_kernelgen_tests.py forbids in
    # tests, so the properties that only the dispatch path used to expose are
    # pinned structurally instead: that a strided narrow source is read by this
    # backend's own kernel rather than by contiguous() (which would re-enter
    # aten::copy_), and that the device-transfer guard refuses rather than
    # recursing. Direct calls are also how TransformerEngine reaches copy_.
    assert api == "direct"
    return flag_gems.copy_(dst, src)


def _filled(shape, dtype, value, device=None):
    """A tensor whose every byte is ``value``, for any dtype including bool."""
    numel = math.prod(shape) if shape else 1
    esize = torch.empty((), dtype=dtype).element_size()
    flat = torch.full((numel * esize,), value, dtype=torch.uint8, device=device)
    return flat.view(dtype).reshape(shape)


def _payload(shape, dtype, seed):
    """Deterministic payload whose bytes are arbitrary but never the sentinel."""
    numel = math.prod(shape) if shape else 1
    esize = torch.empty((), dtype=dtype).element_size()
    torch.manual_seed(seed)
    if dtype is torch.bool:
        return torch.randint(0, 2, (numel,), dtype=torch.bool).reshape(shape)
    raw = torch.randint(0, 256, (numel * esize,), dtype=torch.uint8)
    # 0x5A is the sentinel the destinations are filled with.
    raw[raw == 0x5A] = 0x5B
    return raw.view(dtype).reshape(shape)


def _check_copy(payload_cpu, api, dst_view=lambda x: x, src_view=lambda x: x):
    """Copy ``payload`` into a guarded destination and check the whole storage."""
    dtype = payload_cpu.dtype
    src = src_view(payload_cpu.to(flag_gems.device))
    numel = src.numel()
    base = _filled((numel + 2 * GUARD,), dtype, 0x5A, flag_gems.device)
    dst = dst_view(base[GUARD : GUARD + numel].reshape(src.shape))
    pointer, offset, strides = dst.data_ptr(), dst.storage_offset(), dst.stride()

    result = _run_copy(dst, src, api)
    torch.npu.synchronize()

    assert result is dst
    assert result.data_ptr() == pointer
    assert result.storage_offset() == offset
    assert result.stride() == strides
    _assert_same_bytes(result, src)
    # The guards catch a launch that walked past the tensor it was given. They
    # are compared as bytes, since a sentinel byte is not a valid bool.
    esize = base.element_size()
    raw = base.reshape(-1).view(torch.uint8).cpu()
    guards = torch.cat((raw[: GUARD * esize], raw[-GUARD * esize :]))
    torch.testing.assert_close(guards, torch.full_like(guards, 0x5A), rtol=0, atol=0)


# --------------------------------------------------------------------------
# shapes, dtypes and layouts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (4095,),
        (4096,),
        (4097,),
        (8192,),
        (12289,),
        (3, 5, 1371),
    ],
)
def test_copy_contiguous_boundaries_exact(checked_launches, api, dtype, shape):
    _check_copy(_payload(shape, dtype, 1), api)
    assert checked_launches


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_copy_zero_dim_exact(api, dtype):
    _check_copy(_payload((), dtype, 2).reshape(()), api)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_copy_empty_exact(api, dtype, shape):
    # The shared implementation hands this to PyTorch, and that path ends in
    # aten::_copy_from, which torch_npu does not register.
    src = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    dst = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    result = _run_copy(dst, src, api)
    torch.npu.synchronize()
    assert result is dst
    assert result.numel() == 0
    assert result.shape == torch.Size(shape)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "permute", "step", "slice", "diagonal"]
)
def test_copy_noncontiguous_source_exact(api, dtype, layout):
    # pointwise_dynamic reads the wrong bytes from element 32 on when a 1-byte
    # operand is strided, so these are the cases a shared flat path cannot take.
    payload = _payload((5 * 7 * 11,), dtype, 3)

    def view(x):
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

    assert not view(payload).is_contiguous()
    _check_copy(payload, api, src_view=view)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", ONE_BYTE_DTYPES + [torch.int16, torch.float32])
@pytest.mark.parametrize("step", [2, 3, 7])
def test_copy_strided_source_every_element(api, dtype, step):
    payload = _payload((4096,), dtype, 4)
    _check_copy(payload, api, src_view=lambda x: x[::step])


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", ONE_BYTE_DTYPES + [torch.float32])
@pytest.mark.parametrize("layout", ["step", "transpose"])
def test_copy_noncontiguous_destination_exact(api, dtype, layout):
    numel = 2048 if layout == "step" else 64 * 32
    payload = _payload((numel,), dtype, 5)
    src = payload.to(flag_gems.device)
    if layout == "step":
        base = _filled((numel * 2,), dtype, 0x5A, flag_gems.device)
        dst = base[::2]
    else:
        base = _filled((32, 64), dtype, 0x5A, flag_gems.device)
        dst = base.t()
        src = src.reshape(64, 32)
    assert not dst.is_contiguous()
    result = _run_copy(dst, src, api)
    torch.npu.synchronize()
    _assert_same_bytes(result, src)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dst_dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("src_dtype", ONE_BYTE_DTYPES + [torch.int16, torch.float16])
def test_copy_strided_narrow_source_cross_dtype(api, src_dtype, dst_dtype):
    # A conversion does not go through the flat path, so the strided narrow
    # source reaches pointwise_dynamic unless it is read here first.
    # Finite values for a float source: a random bit pattern is mostly NaN, and
    # a conversion is not required to carry the NaN payload through (this kernel
    # canonicalises it to 0x7fffffff, PyTorch on the host keeps it).
    if src_dtype.is_floating_point:
        full = (torch.arange(4096, dtype=torch.float32) - 2048).to(src_dtype)
    else:
        full = _payload((4096,), src_dtype, 12)
    src = full.to(flag_gems.device)[::2]
    expected = full[::2].to(dst_dtype)
    dst = _filled((2048,), dst_dtype, 0x5A, flag_gems.device)
    result = _run_copy(dst, src, api)
    torch.npu.synchronize()
    _assert_same_bytes(result.cpu(), expected)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", ONE_BYTE_DTYPES + [torch.float16])
def test_copy_broadcast_from_strided_narrow_source(api, dtype):
    # Broadcast does not go through the flat path either.
    full = _payload((2, 256), dtype, 13)
    src = full.to(flag_gems.device)[:1, ::2]
    expected = full[:1, ::2].expand(64, 128)
    dst = _filled((64, 128), dtype, 0x5A, flag_gems.device)
    result = _run_copy(dst, src, api)
    torch.npu.synchronize()
    _assert_same_bytes(result.cpu(), expected)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.bfloat16])
def test_copy_broadcast_source_exact(api, dtype):
    payload = _payload((1, 128), dtype, 6).to(flag_gems.device)
    dst = _filled((64, 128), dtype, 0x5A, flag_gems.device)
    result = _run_copy(dst, payload, api)
    torch.npu.synchronize()
    _assert_same_bytes(result, payload.expand(64, 128))


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_copy_unaligned_storage_offset_exact(api, offset):
    # A byte offset that is not a multiple of 4 is what stops the flat path
    # from reinterpreting the operands as int32.
    payload = _payload((1024 + offset,), torch.int8, 7)
    _check_copy(payload, api, src_view=lambda x: x[offset:])


# --------------------------------------------------------------------------
# values that must survive unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_copy_preserves_special_values_exact(api, dtype):
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
    _check_copy(payload, api)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dst_dtype", NARROW_INT_DTYPES)
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_copy_narrow_int_conversion_matches_cpu(api, dst_dtype, src_dtype):
    # Only values the destination can hold. A float outside the destination's
    # range is undefined in C++ and the platforms disagree: for a float16 32768
    # this kernel gives int16 -32768, the narrowing through int32 that
    # _copy_to_narrow_int_kernel documents, while PyTorch on this aarch64 host
    # gives 32767. Gating on either would be gating on undefined behaviour.
    info = torch.iinfo(dst_dtype)
    values = [-1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5]
    values += [float(info.min), float(info.min) + 1, float(info.max) - 1]
    values += [float(info.max), float(info.max // 2)]
    # Filter after the cast, not before: float16 cannot hold 32767 and rounds it
    # to 32768, which is out of range again.
    rounded = torch.tensor(values, dtype=src_dtype)
    # Compare in float64: in float16 the bound 32767 itself rounds to 32768, so
    # filtering in the source dtype lets the out-of-range value back in.
    exact = rounded.to(torch.float64)
    src = rounded[(exact >= info.min) & (exact <= info.max)]
    expected = src.to(dst_dtype)
    dst = _filled((src.numel(),), dst_dtype, 0x5A, flag_gems.device)
    result = _run_copy(dst, src.to(flag_gems.device), api)
    torch.npu.synchronize()
    _assert_same_bytes(result.cpu(), expected)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.int32, torch.int8])
def test_copy_to_bool_matches_cpu(api, src_dtype):
    values = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=src_dtype)
    expected = values.to(torch.bool)
    dst = _filled((values.numel(),), torch.bool, 0x5A, flag_gems.device)
    result = _run_copy(dst, values.to(flag_gems.device), api)
    torch.npu.synchronize()
    _assert_same_bytes(result.cpu(), expected)


# --------------------------------------------------------------------------
# cases the shared implementation hands back to PyTorch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.float32])
def test_copy_across_devices_exact(dtype):
    # redispatch(CompositeExplicitAutograd) ends in aten::_copy_from, which
    # torch_npu does not register, so both directions have to be served here.
    # This is the direct call, which is how TransformerEngine reaches copy_.
    payload = _payload((1024,), dtype, 8)

    dst = _filled((1024,), dtype, 0x5A, flag_gems.device)
    flag_gems.copy_(dst, payload)
    torch.npu.synchronize()
    _assert_same_bytes(dst, payload.to(flag_gems.device))

    back = _filled((1024,), dtype, 0x5A)
    flag_gems.copy_(back, dst)
    torch.npu.synchronize()
    _assert_same_bytes(back, payload)


def test_copy_across_devices_is_refused_when_re_entered(copy_ops, monkeypatch):
    # A transfer needs the runtime, and every primitive that could perform one
    # -- to, _to_copy, Storage.copy_ -- routes back through aten::copy_. Once
    # copy_ is registered for dispatch there is no way out, which is why it is
    # in CUSTOMIZED_UNUSED_OPS on this backend. Entering the transfer a second
    # time is exactly what that recursion looks like, so pin the refusal: the
    # alternative is a thousand-frame stack overflow.
    monkeypatch.setattr(copy_ops._reentry, "transferring", True, raising=False)
    host = torch.arange(16, dtype=torch.float32)
    dst = torch.zeros(16, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(NotImplementedError, match="between devices"):
        copy_ops.copy_(dst, host)


def test_copy_strided_narrow_source_uses_its_own_kernel(copy_ops, monkeypatch):
    # The source is read by _gather_copy_kernel rather than materialised with
    # contiguous(): contiguous() is empty_like plus copy_, so it re-enters this
    # operator and recurses the moment copy_ is registered for dispatch.
    launched = []
    kernel = copy_ops._gather_copy_kernel

    class CountingKernel:
        def __getitem__(self, grid):
            launched.append(grid)
            return kernel[grid]

    monkeypatch.setattr(copy_ops, "_gather_copy_kernel", CountingKernel())

    def refuse(*args, **kwargs):
        raise AssertionError("copy_ must not call contiguous() on its source")

    monkeypatch.setattr(torch.Tensor, "contiguous", refuse)
    full = _payload((4096,), torch.int8, 19)
    src = full.to(flag_gems.device)[::2]
    dst = _filled((2048,), torch.int8, 0x5A, flag_gems.device)
    copy_ops.copy_(dst, src)
    torch.npu.synchronize()
    assert launched
    assert torch.equal(dst.cpu(), full[::2])


@pytest.mark.parametrize("api", APIS)
def test_copy_overlapping_alias_exact(api):
    base = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    expected = torch.cat(
        (
            torch.zeros(1, dtype=torch.float32),
            torch.arange(15, dtype=torch.float32),
        )
    )
    _run_copy(base[1:], base[:-1], api)
    torch.npu.synchronize()
    _assert_same_bytes(base.cpu(), expected)


@pytest.mark.parametrize("api", APIS)
def test_copy_identical_alias_is_a_no_op(api):
    base = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    expected = torch.arange(16, dtype=torch.float32)
    result = _run_copy(base, base, api)
    torch.npu.synchronize()
    assert result is base
    _assert_same_bytes(base.cpu(), expected)


def test_copy_functional_matches_inplace_exact():
    payload = _payload((4097,), torch.bfloat16, 9).to(flag_gems.device)
    template = _filled((4097,), torch.bfloat16, 0x5A, flag_gems.device)
    out = flag_gems.copy(template, payload)
    torch.npu.synchronize()
    assert out.data_ptr() != template.data_ptr()
    _assert_same_bytes(out, payload)


# copy() allocates with the template's own strides and then calls copy_, so it
# reaches the same paths with a destination the in-place tests never produce:
# one that is not contiguous because the template was not.


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1,), (4095,), (4096,), (3, 5, 1371)])
def test_copy_functional_shapes_exact(dtype, shape):
    payload = _payload(shape, dtype, 15).to(flag_gems.device)
    template = _filled(shape, dtype, 0x5A, flag_gems.device)
    out = flag_gems.copy(template, payload)
    torch.npu.synchronize()
    assert out.dtype == dtype and out.shape == torch.Size(shape)
    assert out.data_ptr() != template.data_ptr()
    _assert_same_bytes(out, payload)


@pytest.mark.parametrize("dtype", ONE_BYTE_DTYPES + [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["transpose", "step"])
def test_copy_functional_strided_template_exact(dtype, layout):
    # empty_strided reproduces the template's strides, so the destination is as
    # non-contiguous as the template is.
    if layout == "transpose":
        template = _filled((32, 64), dtype, 0x5A, flag_gems.device).t()
        payload = _payload((64, 32), dtype, 16).to(flag_gems.device)
    else:
        template = _filled((4096,), dtype, 0x5A, flag_gems.device)[::2]
        payload = _payload((2048,), dtype, 16).to(flag_gems.device)
    assert not template.is_contiguous()
    out = flag_gems.copy(template, payload)
    torch.npu.synchronize()
    assert out.stride() == template.stride()
    _assert_same_bytes(out, payload)


@pytest.mark.parametrize("dtype", ONE_BYTE_DTYPES + [torch.float16])
def test_copy_functional_strided_source_exact(dtype):
    full = _payload((4096,), dtype, 17)
    src = full.to(flag_gems.device)[::2]
    template = _filled((2048,), dtype, 0x5A, flag_gems.device)
    out = flag_gems.copy(template, src)
    torch.npu.synchronize()
    _assert_same_bytes(out.cpu(), full[::2])


@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.bfloat16])
def test_copy_functional_broadcast_exact(dtype):
    payload = _payload((1, 128), dtype, 18).to(flag_gems.device)
    template = _filled((64, 128), dtype, 0x5A, flag_gems.device)
    out = flag_gems.copy(template, payload)
    torch.npu.synchronize()
    _assert_same_bytes(out, payload.expand(64, 128))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8])
def test_copy_functional_cross_dtype_exact(dtype):
    # The template fixes the output dtype, so this is the conversion path.
    values = (torch.arange(256, dtype=torch.float32) - 128).to(torch.float32)
    src = values.to(flag_gems.device)
    template = _filled((256,), dtype, 0x5A, flag_gems.device)
    out = flag_gems.copy(template, src)
    torch.npu.synchronize()
    _assert_same_bytes(out.cpu(), values.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8, torch.bool])
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_copy_functional_empty_exact(dtype, shape):
    src = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    template = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    out = flag_gems.copy(template, src)
    torch.npu.synchronize()
    assert out.numel() == 0 and out.shape == torch.Size(shape)


def test_copy_functional_over_core_dim_exact(checked_launches):
    # The functional form reaches the same bounded launch as copy_.
    numel = (65535 + 1) * 4096 * 2
    pattern = torch.arange(1024, dtype=torch.bfloat16, device=flag_gems.device)
    src = pattern.repeat(numel // 1024)
    template = torch.empty(numel, dtype=torch.bfloat16, device=flag_gems.device)
    out = flag_gems.copy(template, src)
    torch.npu.synchronize()
    # The template only supplied the shape and strides; hold src and out alone.
    del template
    torch.npu.empty_cache()
    # Chunked: torch.equal on a pair this size wants another 4 GiB, which is
    # available on one machine and not on another.
    for start in range(0, numel, 2**24):
        chunk = slice(start, start + 2**24)
        torch.testing.assert_close(out[chunk].cpu(), src[chunk].cpu(), rtol=0, atol=0)
    assert checked_launches


# --------------------------------------------------------------------------
# launch limits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "step", "cross_dtype"])
def test_copy_splits_when_wider_than_pointwise_can_index(
    copy_ops, monkeypatch, api, dtype, layout
):
    # Above 2**31 elements pointwise_dynamic can no longer index the operand, so
    # the copy is halved until it fits. Allocating that is out of the question;
    # lower the threshold instead and check the split lands every element.
    monkeypatch.setattr(copy_ops, "_MAX_POINTWISE_ELEMS", 100)
    if layout == "cross_dtype" and dtype.is_floating_point:
        # Finite values: a conversion need not carry a NaN payload through.
        payload = (torch.arange(37 * 11, dtype=torch.float32) - 200).to(dtype)
        payload = payload.reshape(37, 11)
    else:
        payload = _payload((37, 11), dtype, 14)
    if layout == "cross_dtype":
        src = payload.to(flag_gems.device)
        dst = _filled((37, 11), torch.float32, 0x5A, flag_gems.device)
        result = _run_copy(dst, src, api)
        torch.npu.synchronize()
        _assert_same_bytes(result.cpu(), payload.to(torch.float32))
        return
    view = (lambda x: x) if layout == "contiguous" else (lambda x: x[::2])
    _check_copy(payload, api, src_view=view)


@pytest.mark.parametrize("tiles", [1, 2, 65534, 65535, 65536, 65537, 131072, 524288])
def test_flat_grid_never_exceeds_core_dim(copy_ops, tiles):
    # Planned on the host: the 65536 and 131072 boundaries are reachable here
    # without allocating the tensors that would reach them on the device.
    n_elements = tiles * copy_ops._FLAT_COPY_BLOCK
    grid = copy_ops._flat_grid(n_elements)
    assert (
        0 < math.prod(grid) <= CORE_DIM_LIMIT
    ), f"n_elements={n_elements}, grid={grid}, coreDim must be <= {CORE_DIM_LIMIT}"
    assert math.prod(grid) == min(tiles, CORE_DIM_LIMIT)


@pytest.mark.parametrize("n_elements", [1, 4095, 4096, 4097])
def test_flat_grid_is_never_zero(copy_ops, n_elements):
    # coreDim has a lower bound too: a grid of 0 is rejected as well.
    assert math.prod(copy_ops._flat_grid(n_elements)) >= 1


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_copy_capped_grid_covers_every_tile(
    copy_ops, checked_launches, monkeypatch, api, dtype, cap
):
    # Force the in-kernel loop to go round many times on a small tensor. The
    # guards and the byte comparison together catch both a tile that was never
    # written and a write that landed outside the destination.
    monkeypatch.setattr(copy_ops, "_MAX_CORE_DIM", cap)
    numel = 9 * copy_ops._FLAT_COPY_BLOCK + 257
    _check_copy(_payload((numel,), dtype, 10), api)
    assert checked_launches
    assert all(math.prod(grid) == cap for grid, *_ in checked_launches)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("extra", [0, 7])
def test_flat_kernel_offset_widths_agree(copy_ops, dtype, extra):
    # WIDE_OFFSETS only differs past 2**31 elements, which is too large to
    # allocate; both widths are correct at any size, so compare them directly
    # instead of leaving the wide branch untested.
    numel = (1 << 20) + extra
    payload = _payload((numel,), dtype, 11).to(flag_gems.device)
    results = []
    for wide in (False, True):
        dst = _filled((numel,), dtype, 0x5A, flag_gems.device)
        as_int32, n = copy_ops._flat_plan(dst, payload)
        view = payload.reshape(-1)
        out = dst.reshape(-1)
        if as_int32:
            view = view.view(torch.int32)
            out = out.view(torch.int32)
        copy_ops._flat_copy_kernel[copy_ops._flat_grid(n)](
            view,
            out,
            n,
            copy_ops._FLAT_COPY_BLOCK,
            n % copy_ops._FLAT_COPY_BLOCK != 0,
            wide,
        )
        torch.npu.synchronize()
        results.append(dst)
        _assert_same_bytes(dst, payload)
    _assert_same_bytes(results[0], results[1])


@pytest.mark.parametrize(
    "numel,dtype",
    [
        (65535 * 4096 * 2, torch.bfloat16),  # the last grid the driver accepts
        (65536 * 4096 * 2, torch.bfloat16),  # one tile past it
        (65536 * 4096 + 17, torch.int32),  # past it, with a tail
    ],
)
def test_copy_real_large_input_exact(checked_launches, numel, dtype):
    # The sizes that made the shared flat path fail with EE1003 coreDim.
    pattern = torch.arange(1024, dtype=dtype, device=flag_gems.device)
    src = pattern.repeat(numel // 1024 + 1)[:numel]
    dst = _filled((numel,), dtype, 0x5A, flag_gems.device)

    result = flag_gems.copy_(dst, src)
    torch.npu.synchronize()

    assert result is dst
    # Compare every element, with bounded host-memory use.
    for start in range(0, numel, 2**24):
        chunk = slice(start, start + 2**24)
        torch.testing.assert_close(
            result[chunk].cpu(), src[chunk].cpu(), rtol=0, atol=0, equal_nan=True
        )
    assert checked_launches
