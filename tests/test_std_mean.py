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

import warnings

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

_COMPLEX32_UNSUPPORTED_VENDORS = {"tsingmicro"}


def _has_std_mean_overload(overload):
    """Query the installed vendor torch; its ATen surface may differ by release."""
    try:
        getattr(torch.ops.aten.std_mean, overload)
    except (AttributeError, RuntimeError):
        return False
    return True


def _missing_schema_reason(overload):
    return f"aten::std_mean.{overload} is unavailable in torch {torch.__version__}"


def _dtype_parameters():
    if cfg.QUICK_MODE:
        return [torch.float32, torch.complex64]

    parameters = [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not utils.bf16_is_supported,
                reason="the backend does not support bfloat16",
            ),
        ),
        torch.float32,
        pytest.param(
            torch.float64,
            marks=pytest.mark.skipif(
                not utils.fp64_is_supported and flag_gems.vendor_name != "mthreads",
                reason="the backend does not support float64",
            ),
        ),
        torch.complex64,
        pytest.param(
            torch.complex128,
            marks=pytest.mark.skipif(
                not utils.fp64_is_supported and flag_gems.vendor_name != "mthreads",
                reason="the backend does not support complex128",
            ),
        ),
    ]
    complex32 = getattr(torch, "complex32", None)
    if complex32 is not None:
        parameters.insert(
            4,
            pytest.param(
                complex32,
                marks=pytest.mark.skipif(
                    flag_gems.vendor_name in _COMPLEX32_UNSUPPORTED_VENDORS,
                    reason="the backend does not support complex32 tensors",
                ),
            ),
        )
    return parameters


STD_MEAN_DTYPES = _dtype_parameters()
SPECIAL_DTYPES = [torch.float32, torch.complex64]
ROW_DTYPES = (
    [torch.float32]
    if cfg.QUICK_MODE
    else [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not utils.bf16_is_supported,
                reason="the backend does not support bfloat16",
            ),
        ),
        torch.float32,
    ]
)
ROW_SHAPES = (
    [(37, 129)]
    if cfg.QUICK_MODE
    else [(64, 64), (4096, 4096), (64, 512, 512), (37, 129)]
)
ROW_CORRECTIONS = [0.5] if cfg.QUICK_MODE else [0.5, -0.5, 1.0]

if cfg.QUICK_MODE:
    DIM_CASES = [((7, 11), -1, False, True)]
    CORRECTION_CASES = [((5, 7, 11), [], 0.5, False, False)]
    OUT_CASES = [((5, 7), [1], 0.5, False, False)]
else:
    DIM_CASES = [
        ((7, 11), 1, True, False),
        ((7, 11), -1, False, True),
        ((3, 5, 7), (0, 2), True, False),
        ((3, 5, 7), [2, 0], False, True),
    ]
    CORRECTION_CASES = [
        ((4097,), None, None, False, False),
        ((5, 7, 11), [], 0.5, False, False),
        ((5, 7, 11), [-1], 0.25, True, False),
        ((5, 7, 11), [0, 2], -2.5, False, False),
        ((5, 7, 11), [0, -1], 1.5, True, True),
    ]
    OUT_CASES = [
        ((257,), None, 1.0, False, False),
        ((5, 7, 11), [], -0.5, False, False),
        ((5, 7, 11), [0, 2], 0.5, True, True),
    ]


def _make_input(shape, dtype, noncontiguous=False):
    input_shape = shape
    if noncontiguous:
        assert len(shape) >= 2
        input_shape = (shape[1], shape[0], *shape[2:])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ComplexHalf support is experimental.*",
            category=UserWarning,
        )
        if dtype.is_complex:
            real_dtype = _std_dtype(dtype)
            real = torch.randn(input_shape, dtype=real_dtype, device=flag_gems.device)
            imag = torch.randn(input_shape, dtype=real_dtype, device=flag_gems.device)
            inp = torch.complex(real, imag)
        else:
            inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)

    if noncontiguous:
        inp = inp.transpose(0, 1)
        assert inp.shape == shape
        assert not inp.is_contiguous()
    return inp


def _reference_input(inp):
    inp = inp.rename(None)
    if flag_gems.vendor_name == "ascend" and inp.dtype == getattr(
        torch, "complex32", None
    ):
        # The NPU copy kernel rejects noncontiguous complex32 tensors. Copy
        # their real components before constructing the high-precision reference.
        components = utils.to_reference(torch.view_as_real(inp), upcast=True)
        return torch.view_as_complex(components)
    return utils.to_reference(inp, upcast=True)


def _std_dtype(dtype):
    if dtype == getattr(torch, "complex32", None):
        return torch.float16
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def _without_names(tensor):
    if any(name is not None for name in tensor.names):
        return tensor.rename(None)
    return tensor


def _assert_std_mean(result, reference, dtype, *, equal_nan=False, atol=1e-4):
    result_std, result_mean = result
    reference_std, reference_mean = reference
    std_dtype = _std_dtype(dtype)

    assert result_std.dtype == std_dtype
    assert result_mean.dtype == dtype
    utils.gems_assert_close(
        _without_names(result_std),
        _without_names(reference_std),
        std_dtype,
        equal_nan=equal_nan,
        atol=atol,
    )
    utils.gems_assert_close(
        _without_names(result_mean),
        _without_names(reference_mean),
        dtype,
        equal_nan=equal_nan,
        atol=atol,
    )


def _output_shape(shape, dim, keepdim):
    ndim = len(shape)
    if dim is None or dim == [] or dim == ():
        dims = set(range(ndim))
    else:
        dim_list = [dim] if isinstance(dim, int) else list(dim)
        dims = {value % ndim for value in dim_list}

    if keepdim:
        return tuple(1 if index in dims else size for index, size in enumerate(shape))
    return tuple(size for index, size in enumerate(shape) if index not in dims)


@pytest.mark.std_mean
@pytest.mark.skipif(
    not _has_std_mean_overload("default"),
    reason=_missing_schema_reason("default"),
)
@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
def test_std_mean(unbiased, dtype):
    inp = _make_input((257, 33), dtype)
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, unbiased=unbiased)
    result = flag_gems.std_mean(inp, unbiased=unbiased)

    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_dim
@pytest.mark.skipif(
    not _has_std_mean_overload("dim"),
    reason=_missing_schema_reason("dim"),
)
@pytest.mark.parametrize("shape,dim,unbiased,keepdim", DIM_CASES)
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
def test_std_mean_dim(shape, dim, unbiased, keepdim, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, dim=dim, unbiased=unbiased, keepdim=keepdim)
    result = flag_gems.std_mean_dim(inp, dim=dim, unbiased=unbiased, keepdim=keepdim)

    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("shape,dim,correction,keepdim,noncontiguous", CORRECTION_CASES)
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
def test_std_mean_correction(shape, dim, correction, keepdim, noncontiguous, dtype):
    inp = _make_input(shape, dtype, noncontiguous)
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, dim=dim, correction=correction, keepdim=keepdim)
    result = flag_gems.std_mean_correction(
        inp, dim=dim, correction=correction, keepdim=keepdim
    )

    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_names_dim
@pytest.mark.skipif(
    not _has_std_mean_overload("names_dim"),
    reason=_missing_schema_reason("names_dim"),
)
@pytest.mark.parametrize(
    "named_dim,numeric_dim,unbiased,keepdim",
    [
        ("reduce", 1, False, False),
        (("batch", "feature"), (0, 2), True, True),
    ],
)
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
def test_std_mean_names_dim(named_dim, numeric_dim, unbiased, keepdim, dtype):
    inp = _make_input((5, 7, 11), dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Named tensors and all their associated APIs are an experimental feature.*",
            category=UserWarning,
        )
        inp = inp.refine_names("batch", "reduce", "feature")

    ref_inp = _reference_input(inp)
    reference = torch.std_mean(
        ref_inp, dim=numeric_dim, unbiased=unbiased, keepdim=keepdim
    )
    # Named tensors cannot cross a Python torch.library backend kernel on the
    # supported torch releases. Exercise the named overload implementation
    # directly, as the existing named-dimension reduction tests do.
    result = flag_gems.std_mean_names_dim(
        inp, dim=named_dim, unbiased=unbiased, keepdim=keepdim
    )

    reduced_dims = {numeric_dim} if isinstance(numeric_dim, int) else set(numeric_dim)
    expected_names = tuple(
        name
        for index, name in enumerate(inp.names)
        if keepdim or index not in reduced_dims
    )
    assert result[0].names == expected_names
    assert result[1].names == expected_names
    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_correction_names
@pytest.mark.skipif(
    not _has_std_mean_overload("correction_names"),
    reason=_missing_schema_reason("correction_names"),
)
@pytest.mark.parametrize(
    "named_dim,numeric_dim,correction,keepdim",
    [
        ("feature", 2, 0.5, True),
        (("batch", "feature"), (0, 2), -1.5, False),
    ],
)
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
def test_std_mean_correction_names(dtype, named_dim, numeric_dim, correction, keepdim):
    inp = _make_input((5, 7, 11), dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Named tensors and all their associated APIs are an experimental feature.*",
            category=UserWarning,
        )
        inp = inp.refine_names("batch", "reduce", "feature")

    ref_inp = _reference_input(inp)
    reference = torch.std_mean(
        ref_inp, dim=numeric_dim, correction=correction, keepdim=keepdim
    )
    result = flag_gems.std_mean_correction_names(
        inp, dim=named_dim, correction=correction, keepdim=keepdim
    )

    reduced_dims = {numeric_dim} if isinstance(numeric_dim, int) else set(numeric_dim)
    expected_names = tuple(
        name
        for index, name in enumerate(inp.names)
        if keepdim or index not in reduced_dims
    )
    assert result[0].names == expected_names
    assert result[1].names == expected_names
    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_correction_out
@pytest.mark.skipif(
    not _has_std_mean_overload("correction_out"),
    reason=_missing_schema_reason("correction_out"),
)
@pytest.mark.parametrize("shape,dim,correction,keepdim,noncontiguous", OUT_CASES)
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
def test_std_mean_correction_out(shape, dim, correction, keepdim, noncontiguous, dtype):
    inp = _make_input(shape, dtype, noncontiguous)
    ref_inp = _reference_input(inp)
    output_shape = _output_shape(shape, dim, keepdim)
    std_out = torch.empty(
        output_shape, dtype=_std_dtype(dtype), device=flag_gems.device
    )
    mean_out = torch.empty(output_shape, dtype=dtype, device=flag_gems.device)

    reference = torch.std_mean(ref_inp, dim=dim, correction=correction, keepdim=keepdim)
    result = flag_gems.std_mean_correction_out(
        inp,
        dim,
        correction=correction,
        keepdim=keepdim,
        out0=std_out,
        out1=mean_out,
    )

    assert result[0] is std_out
    assert result[1] is mean_out
    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", SPECIAL_DTYPES)
@pytest.mark.parametrize("shape,dim", [((2, 0, 3), [1]), ((0, 3), [])])
def test_std_mean_empty_reduction(shape, dim, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reference = torch.std_mean(ref_inp, dim=dim, correction=0)
    with pytest.warns(UserWarning, match="degrees of freedom is <= 0"):
        result = flag_gems.std_mean_correction(inp, dim=dim, correction=0)

    _assert_std_mean(result, reference, dtype, equal_nan=True)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
def test_std_mean_empty_negative_correction_does_not_warn():
    inp = _make_input((2, 0, 3), torch.float32)
    ref_inp = _reference_input(inp)
    reference = torch.std_mean(ref_inp, dim=1, correction=-0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = flag_gems.std_mean_correction(inp, dim=1, correction=-0.5)

    assert not any("degrees of freedom is <= 0" in str(item.message) for item in caught)
    _assert_std_mean(result, reference, torch.float32, equal_nan=True)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", SPECIAL_DTYPES)
@pytest.mark.parametrize("correction", [-0.5, 0.0, 1.0])
def test_std_mean_empty_output_warning(dtype, correction):
    inp = _make_input((0, 3), dtype)
    with warnings.catch_warnings(record=True) as native_warnings:
        warnings.simplefilter("always")
        reference = torch.std_mean(_reference_input(inp), dim=1, correction=correction)
    with warnings.catch_warnings(record=True) as gems_warnings:
        warnings.simplefilter("always")
        result = flag_gems.std_mean_correction(inp, dim=1, correction=correction)

    message = "degrees of freedom is <= 0"
    native_dof = any(message in str(item.message) for item in native_warnings)
    gems_dof = any(message in str(item.message) for item in gems_warnings)
    assert native_dof == gems_dof
    assert result[0].dtype == _std_dtype(dtype)
    assert result[1].dtype == dtype
    for actual, expected in zip(result, reference):
        assert actual.shape == expected.shape == (0,)
        assert actual.numel() == 0


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", SPECIAL_DTYPES)
@pytest.mark.parametrize(
    "shape,dim,correction",
    [((1,), None, 1.0), ((2,), None, 2.0), ((2, 3), [1], 3.5)],
)
def test_std_mean_warns_for_nonpositive_dof(shape, dim, correction, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = _reference_input(inp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reference = torch.std_mean(ref_inp, dim=dim, correction=correction)
    with pytest.warns(UserWarning, match="degrees of freedom is <= 0"):
        result = flag_gems.std_mean_correction(inp, dim=dim, correction=correction)

    _assert_std_mean(result, reference, dtype, equal_nan=True)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
def test_std_mean_complex_nonpositive_dof_is_componentwise():
    inp = torch.tensor(
        [1.0 + 0.0j, 3.0 + 0.0j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    ref_inp = _reference_input(inp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reference = torch.std_mean(ref_inp, correction=2.0)
    with pytest.warns(UserWarning, match="degrees of freedom is <= 0"):
        result = flag_gems.std_mean_correction(inp, correction=2.0)

    _assert_std_mean(
        result,
        reference,
        torch.complex64,
        equal_nan=True,
    )


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize(
    "dim,error_type",
    [([1, -2], RuntimeError), ([3], IndexError), ([-4], IndexError)],
)
def test_std_mean_dim_errors(dim, error_type):
    inp = _make_input((3, 5, 7), torch.float32)

    with pytest.raises(error_type):
        flag_gems.std_mean_correction(inp, dim=dim, correction=0.5)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", SPECIAL_DTYPES)
def test_std_mean_numerical_stability(dtype):
    index = torch.arange(8192, dtype=torch.float32, device=flag_gems.device)
    real = index.remainder(7).sub(3).add(10000)
    if dtype == torch.complex64:
        imag = index.remainder(5).sub(2).sub(12000)
        inp = torch.complex(real, imag)
    else:
        inp = real
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, correction=0.5)
    result = flag_gems.std_mean_correction(inp, correction=0.5)

    _assert_std_mean(result, reference, dtype, atol=2e-3)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", SPECIAL_DTYPES)
def test_std_mean_numerical_stability_with_first_outlier(dtype):
    real = torch.zeros(65536, dtype=torch.float32, device=flag_gems.device)
    real[0] = 1.0e8
    if dtype == torch.complex64:
        imag = torch.ones_like(real)
        imag[0] = -1.0e8
        inp = torch.complex(real, imag)
    else:
        inp = real
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, correction=-0.5)
    result = flag_gems.std_mean_correction(inp, correction=-0.5)

    _assert_std_mean(result, reference, dtype, atol=2e-3)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
def test_std_mean_large_global_reduction():
    size = 512 * 1024 + 1
    index = torch.arange(size, dtype=torch.float32, device=flag_gems.device)
    inp = index.remainder(17).sub(8).add(10000)
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, correction=-0.25)
    result = flag_gems.std_mean_correction(inp, correction=-0.25)

    _assert_std_mean(result, reference, torch.float32, atol=2e-3)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not utils.bf16_is_supported,
                reason="the backend does not support bfloat16",
            ),
        ),
    ],
)
def test_std_mean_low_precision_accumulates_in_float32(dtype):
    inp = _make_input((8192,), dtype).mul_(2).add_(100)
    ref_inp = _reference_input(inp)

    reference = torch.std_mean(ref_inp, correction=-0.25)
    result = flag_gems.std_mean_correction(inp, correction=-0.25)

    assert torch.isfinite(result[0]).item()
    assert torch.isfinite(result[1]).item()
    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", ROW_DTYPES)
@pytest.mark.parametrize("shape", ROW_SHAPES)
@pytest.mark.parametrize("correction", ROW_CORRECTIONS)
def test_std_mean_rows_core_and_tail(dtype, shape, correction):
    inp = _make_input(shape, dtype)
    reference = torch.std_mean(_reference_input(inp), dim=-1, correction=correction)
    result = flag_gems.std_mean_correction(inp, [-1], correction=correction)

    _assert_std_mean(result, reference, dtype, atol=2e-3)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES)
@pytest.mark.parametrize("shape", [(64, 64), (37, 129)])
def test_std_mean_row_dtypes(dtype, shape):
    inp = _make_input(shape, dtype)
    reference = torch.std_mean(_reference_input(inp), dim=-1, correction=0.5)
    result = flag_gems.std_mean_correction(inp, [-1], correction=0.5)

    _assert_std_mean(result, reference, dtype)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", ROW_DTYPES)
@pytest.mark.parametrize("case", ["offset", "outlier"])
def test_std_mean_rows_stability(dtype, case):
    if case == "offset":
        index = torch.arange(4096, dtype=torch.float32, device=flag_gems.device)
        offset = 1.0e6 if dtype == torch.float32 else 1000.0
        inp = index.remainder(7).sub(3).add(offset).repeat(37, 1).to(dtype)
    else:
        inp = torch.zeros((37, 4096), dtype=dtype, device=flag_gems.device)
        inp[:, 0] = 60000
    reference = torch.std_mean(_reference_input(inp), dim=-1, correction=0.5)
    result = flag_gems.std_mean_correction(inp, [-1], correction=0.5)

    _assert_std_mean(result, reference, dtype, atol=2e-3)


@pytest.mark.std_mean_correction
@pytest.mark.skipif(
    not _has_std_mean_overload("correction"),
    reason=_missing_schema_reason("correction"),
)
@pytest.mark.parametrize("dtype", ROW_DTYPES)
@pytest.mark.parametrize("case", ["nan", "inf", "negative_inf", "mixed_inf"])
def test_std_mean_rows_nonfinite(dtype, case):
    original = torch.ones((37, 129), dtype=dtype)
    original[:, 0] = float("nan") if case == "nan" else float("inf")
    if case == "negative_inf":
        original[:, 0] = -float("inf")
    elif case == "mixed_inf":
        original[:, 1] = -float("inf")
    inp = original.to(flag_gems.device)
    std, mean = flag_gems.std_mean_correction(inp, [-1], correction=0.5)

    assert std.dtype == dtype
    assert mean.dtype == dtype
    assert torch.isnan(std.cpu()).all()
    mean = mean.cpu()
    if case in ("nan", "mixed_inf"):
        assert torch.isnan(mean).all()
    else:
        # Native Welford's mean for one-signed infinity can be Inf or NaN,
        # depending on reduction order; it must never become a finite value.
        signed_inf = (
            torch.isneginf(mean) if case == "negative_inf" else torch.isposinf(mean)
        )
        assert (torch.isnan(mean) | signed_inf).all()
