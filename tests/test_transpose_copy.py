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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_BASE_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8, torch.bool]
)
TRANSPOSE_COPY_DTYPES = list(dict.fromkeys(_BASE_DTYPES))


def _float8_dtypes():
    dtype_names = [
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ]
    dtype_params = []
    for name in dtype_names:
        if not hasattr(torch, name):
            continue
        dtype = getattr(torch, name)
        try:
            input_probe = torch.empty(1, dtype=torch.uint8, device=flag_gems.device)
            input_probe.view(dtype)
            output_probe = torch.empty(1, dtype=dtype, device=flag_gems.device)
            output_probe.view(torch.uint8)
        except (NotImplementedError, RuntimeError, TypeError) as error:
            if not any(
                text in str(error).lower()
                for text in ("not support", "unsupported", "not implemented")
            ):
                raise
            dtype_params.append(
                pytest.param(
                    dtype,
                    marks=pytest.mark.skip(
                        reason=(
                            f"{flag_gems.vendor_name} does not support {dtype} "
                            f"byte storage views: {error}"
                        )
                    ),
                )
            )
        else:
            dtype_params.append(dtype)
    return dtype_params


FLOAT8_DTYPES = _float8_dtypes()

TRANSPOSE_COPY_CASES = [
    ((7,), 0, 0),
    ((2, 3), 0, 1),
    ((2, 3, 5), 0, -1),
    ((2, 3, 4, 5), -3, -1),
    ((1, 7, 1), 1, -1),
    ((0, 3, 5), 0, 2),
]

TRANSPOSE_COPY_FLOAT8_CASES = [
    ((), 0, 0, False),
    ((7,), 0, 0, False),
    ((4, 6), 0, 1, False),
    ((2, 3, 4), 0, -1, False),
    ((2, 3, 4, 5), 1, -1, False),
    ((2, 3, 4), 1, 1, False),
    ((2, 3, 5), 0, -1, True),
]


def _make_input(shape, dtype, device):
    numel = math.prod(shape)
    if dtype == torch.bool:
        values = torch.arange(numel, dtype=torch.int64) % 2 == 0
    elif dtype.is_complex:
        real = torch.arange(numel, dtype=torch.float32) - numel // 2
        values = torch.complex(real, real + 1).to(dtype)
    elif dtype.is_floating_point:
        values = torch.arange(numel, dtype=torch.float32).to(dtype)
    else:
        values = torch.arange(numel, dtype=torch.int64).to(dtype)
    return values.reshape(shape).to(device)


def _complex_params():
    params = []
    for name in ("complex32", "complex64", "complex128"):
        if not hasattr(torch, name):
            continue
        dtype = getattr(torch, name)
        for metadata in ("plain", "conj", "neg", "conj_neg"):
            marks = []
            try:
                # Byte transfers avoid native complex arithmetic and FP64 casts.
                probe_bytes = torch.zeros((2, 3), dtype=dtype).view(torch.uint8)
                probe = probe_bytes.to(flag_gems.device).view(dtype)
                torch.empty((2, 3), dtype=dtype, device=flag_gems.device).view(
                    torch.uint8
                )
                probe.view(torch.uint8).cpu()
                if "conj" in metadata:
                    probe = probe.conj()
                if "neg" in metadata:
                    probe = torch._neg_view(probe)
            except (NotImplementedError, RuntimeError, TypeError) as error:
                message = str(error).lower()
                if not any(
                    text in message
                    for text in ("not support", "unsupported", "not implemented")
                ):
                    raise
                marks.append(
                    pytest.mark.skip(
                        reason=f"{flag_gems.vendor_name}: {name}/{metadata}: {error}"
                    )
                )
            params.append(
                pytest.param(dtype, metadata, marks=marks, id=f"{name}-{metadata}")
            )
    return params


COMPLEX_PARAMS = _complex_params()
COMPLEX_CASES = [
    pytest.param((), 0, 0, "plain", id="scalar"),
    pytest.param((17,), 0, -1, "plain", id="vector"),
    pytest.param((35, 67), 0, 1, "plain", id="tile-tail"),
    pytest.param((2, 3, 5), 0, -1, "plain", id="3d"),
    pytest.param((2, 3, 4, 5), -3, -1, "plain", id="4d"),
    pytest.param((2, 3, 4, 5, 6), 1, -1, "plain", id="5d"),
    pytest.param((2, 2, 2, 2, 2, 2, 2, 2), 0, -1, "plain", id="8d"),
    pytest.param((3, 5), 0, 0, "plain", id="same-dim"),
    pytest.param((3, 5), 0, 1, "transpose", id="transposed"),
    pytest.param((3, 10), 0, 1, "slice", id="last-stride"),
    pytest.param((3, 10), 1, 1, "slice", id="same-dim-strided"),
    pytest.param((17,), 0, 0, "slice", id="strided-vector"),
    pytest.param((3, 5), 0, 1, "offset", id="storage-offset"),
    pytest.param((1, 5), 0, 1, "expand", id="zero-stride"),
    pytest.param((0, 3), 0, 1, "plain", id="empty"),
]


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape,dim0,dim1,layout", COMPLEX_CASES)
@pytest.mark.parametrize("dtype,metadata", COMPLEX_PARAMS)
def test_accuracy_transpose_copy_complex(shape, dim0, dim1, layout, dtype, metadata):
    real_dtype = {
        torch.complex32: torch.float16,
        torch.complex64: torch.float32,
        torch.complex128: torch.float64,
    }[dtype]
    components = (
        torch.arange(math.prod(shape) * 2, dtype=torch.float64) % 127 - 63
    ).to(real_dtype)
    special = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan"), -float("nan")],
        dtype=real_dtype,
    )
    count = min(components.numel(), special.numel())
    components[:count] = special[:count]
    ref_input = torch.view_as_complex(components.reshape(*shape, 2))
    input = (
        ref_input.reshape(-1)
        .view(torch.uint8)
        .to(flag_gems.device)
        .view(dtype)
        .reshape(shape)
    )

    def apply_views(tensor, with_metadata=True):
        if layout == "transpose":
            tensor = tensor.transpose(0, 1)
        elif layout == "slice":
            tensor = tensor[..., ::2]
        elif layout == "offset":
            tensor = tensor[1:, 1:]
        elif layout == "expand":
            tensor = tensor.expand(3, 5)
        if with_metadata and "conj" in metadata:
            tensor = tensor.conj()
        if with_metadata and "neg" in metadata:
            tensor = torch._neg_view(tensor)
        return tensor

    input = apply_views(input)
    # Resolve logical signs component-wise; native complex negation can canonicalize
    # signed zero/NaNs or be unavailable for complex32 with both metadata bits.
    reference = torch.view_as_real(
        apply_views(ref_input, with_metadata=False).transpose(dim0, dim1)
    ).clone(memory_format=torch.contiguous_format)
    if "neg" in metadata:
        reference[..., 0] = -reference[..., 0]
    if ("conj" in metadata) != ("neg" in metadata):
        reference[..., 1] = -reference[..., 1]
    reference = torch.view_as_complex(reference)
    result = flag_gems.transpose_copy(input, dim0, dim1)

    assert result.dtype == dtype
    assert result.shape == reference.shape
    assert result.is_contiguous()
    assert not result.is_conj() and not result.is_neg()
    assert not torch._C._is_alias_of(input, result)
    # Compare storage bits, including NaN payloads and signed zero, on the CPU.
    assert torch.equal(
        result.reshape(-1).view(torch.uint8).cpu(),
        reference.reshape(-1).view(torch.uint8),
    )


def _assert_copy_layout(result, reference, input):
    utils.gems_assert_equal(result, reference)
    assert result.shape == reference.shape
    assert result.stride() == reference.stride()
    assert result.is_contiguous()
    assert not torch._C._is_alias_of(input, result)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape,dim0,dim1", TRANSPOSE_COPY_CASES)
@pytest.mark.parametrize("dtype", TRANSPOSE_COPY_DTYPES)
def test_accuracy_transpose_copy(shape, dim0, dim1, dtype):
    input = _make_input(shape, dtype, flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, dim0, dim1)

    result = flag_gems.transpose_copy(input, dim0, dim1)

    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("dim0,dim1", [(0, 0), (-1, 0), (0, -1), (-1, -1)])
def test_accuracy_transpose_copy_scalar(dim0, dim1):
    input = torch.tensor(3.0, device=flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, dim0, dim1)

    result = flag_gems.transpose_copy(input, dim0, dim1)

    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("dtype", TRANSPOSE_COPY_DTYPES)
def test_accuracy_transpose_copy_non_contiguous(dtype):
    base = _make_input((4, 3, 5), dtype, flag_gems.device)
    input = base[::2]
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 0, -1)

    result = flag_gems.transpose_copy(input, 0, -1)

    assert not input.is_contiguous()
    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
def test_accuracy_transpose_copy_same_dim_does_not_alias():
    input = _make_input((2, 3, 4), torch.float32, flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 1, 1)

    result = flag_gems.transpose_copy(input, 1, 1)

    _assert_copy_layout(result, reference, input)
    assert result.data_ptr() != input.data_ptr()


@pytest.mark.transpose_copy
@pytest.mark.parametrize(
    "shape,dim0,dim1",
    [
        ((), 1, 0),
        ((), -2, 0),
        ((2, 3), 0, 2),
        ((2, 3), -3, 0),
    ],
)
def test_transpose_copy_invalid_dims(shape, dim0, dim1):
    input = _make_input(shape, torch.float32, flag_gems.device)

    with pytest.raises(IndexError, match="Dimension out of range"):
        flag_gems.transpose_copy(input, dim0, dim1)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape,dim0,dim1,non_contiguous", TRANSPOSE_COPY_FLOAT8_CASES)
@pytest.mark.parametrize("dtype", FLOAT8_DTYPES)
def test_accuracy_transpose_copy_float8(shape, dim0, dim1, non_contiguous, dtype):
    base_shape = (shape[0] * 2, *shape[1:]) if non_contiguous else shape
    input_bytes = torch.arange(
        math.prod(base_shape), dtype=torch.uint8, device=flag_gems.device
    ).reshape(base_shape)
    if non_contiguous:
        input_bytes = input_bytes[::2]
    input = (
        input_bytes.reshape(1).view(dtype).reshape(())
        if input_bytes.ndim == 0
        else input_bytes.view(dtype)
    )
    reference = (
        utils.to_reference(input_bytes)
        .transpose(dim0, dim1)
        .clone(memory_format=torch.contiguous_format)
    )

    result = flag_gems.transpose_copy(input, dim0, dim1)

    _assert_copy_layout(result.view(torch.uint8), reference.view(torch.uint8), input)
