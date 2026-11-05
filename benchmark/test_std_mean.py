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
import warnings

import pytest
import torch
import triton
from packaging.version import Version

import flag_gems
from flag_gems.ops.std_mean import (
    std_mean,
    std_mean_correction,
    std_mean_correction_names,
    std_mean_correction_out,
    std_mean_dim,
    std_mean_names_dim,
)

from . import base

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend"
    and Version(str(triton.__version__).split("+", 1)[0]) < Version("3.5"),
    reason="CANN850 (legacy Triton) is validated for accuracy only",
)

_COMPLEX32_UNSUPPORTED_VENDORS = {"tsingmicro"}
_STD_MEAN_OVERLOADS = {
    "std_mean": "default",
    "std_mean_dim": "dim",
    "std_mean_correction": "correction",
    "std_mean_names_dim": "names_dim",
    "std_mean_correction_names": "correction_names",
    "std_mean_correction_out": "correction_out",
}
_STD_MEAN_IMPLS = {
    "std_mean": std_mean,
    "std_mean_dim": std_mean_dim,
    "std_mean_correction": std_mean_correction,
    "std_mean_names_dim": std_mean_names_dim,
    "std_mean_correction_names": std_mean_correction_names,
    "std_mean_correction_out": std_mean_correction_out,
}


def _get_std_mean_op(variant):
    """Resolve against the installed vendor torch instead of assuming a version."""
    try:
        return getattr(torch.ops.aten.std_mean, _STD_MEAN_OVERLOADS[variant])
    except (AttributeError, RuntimeError):
        return None


_STD_MEAN_OPS = {variant: _get_std_mean_op(variant) for variant in _STD_MEAN_OVERLOADS}


def _dtype_parameters(variant, *, named=False):
    def parameter(dtype, identifier, *marks):
        parameter_marks = list(marks)
        if dtype.is_complex:
            parameter_marks.append(
                pytest.mark.skipif(
                    flag_gems.vendor_name == "ascend",
                    reason=(
                        "the Ascend native std_mean baseline rejects complex tensors"
                    ),
                )
            )
        if dtype == torch.float64:
            parameter_marks.append(
                pytest.mark.skipif(
                    flag_gems.vendor_name == "mthreads",
                    reason=(
                        "the MThreads native std_mean baseline illegally "
                        "accesses memory for float64 benchmark shapes"
                    ),
                )
            )
        if dtype == torch.complex128:
            parameter_marks.append(
                pytest.mark.skipif(
                    flag_gems.vendor_name == "mthreads",
                    reason=(
                        "the MThreads native std_mean baseline cannot execute "
                        "complex128 reliably"
                    ),
                )
            )
        if named and dtype.is_complex:
            parameter_marks.append(
                pytest.mark.skip(
                    reason=(
                        "the native std_mean baseline cannot call view_as_real "
                        "on a named complex tensor"
                    )
                )
            )
        return pytest.param(dtype, id=identifier, marks=parameter_marks)

    parameters = [
        parameter(torch.float16, "f16"),
        parameter(
            torch.bfloat16,
            "b16",
            pytest.mark.skipif(
                not flag_gems.runtime.device.support_bf16,
                reason="the backend does not support bfloat16",
            ),
        ),
        parameter(torch.float32, "f32"),
        parameter(
            torch.float64,
            "f64",
            pytest.mark.skipif(
                not flag_gems.runtime.device.support_fp64
                and flag_gems.vendor_name != "mthreads",
                reason="the backend does not support float64",
            ),
        ),
        parameter(torch.complex64, "c64"),
        parameter(
            torch.complex128,
            "c128",
            pytest.mark.skipif(
                not flag_gems.runtime.device.support_fp64
                and flag_gems.vendor_name != "mthreads",
                reason="the backend does not support complex128",
            ),
        ),
    ]

    complex32 = getattr(torch, "complex32", None)
    if complex32 is not None:
        parameters.insert(
            4,
            parameter(
                complex32,
                "c32",
                pytest.mark.skipif(
                    flag_gems.vendor_name in _COMPLEX32_UNSUPPORTED_VENDORS,
                    reason="the backend does not support complex32 tensors",
                ),
            ),
        )
    return parameters


STD_MEAN_DTYPES = {
    variant: _dtype_parameters(
        variant,
        named=variant in {"std_mean_names_dim", "std_mean_correction_names"},
    )
    for variant in _STD_MEAN_OVERLOADS
}


def _make_input(shape, dtype, device):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="ComplexHalf support is experimental.*",
            category=UserWarning,
        )
        if dtype.is_complex:
            real_dtype = _std_dtype(dtype)
            real = torch.randn(shape, dtype=real_dtype, device=device)
            imag = torch.randn(shape, dtype=real_dtype, device=device)
            return torch.complex(real, imag)
        return torch.randn(shape, dtype=dtype, device=device)


def _std_dtype(dtype):
    if dtype == getattr(torch, "complex32", None):
        return torch.float16
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


class StdMeanBenchmark(base.UnaryReductionBenchmark):
    MAX_ELEMENTS = 1 << 24

    def __init__(self, variant, dtype):
        schema_op = _STD_MEAN_OPS[variant]
        if schema_op is None:
            pytest.skip(
                f"aten::std_mean.{_STD_MEAN_OVERLOADS[variant]} is not "
                f"available in torch {torch.__version__}"
            )
        torch_op = schema_op
        # Benchmark the FlagGems implementation directly, as other kernel
        # benchmarks do. Global operator registration can include unrelated
        # Python dispatch gaps in short device timings.
        # Accuracy tests separately exercise all six public FlagGems entry points.
        gems_op = _STD_MEAN_IMPLS[variant]
        if variant == "std_mean_names_dim":
            # Direct aten named calls are rejected by the Python boxed bridge;
            # the public native API and direct FlagGems implementation are the
            # corresponding measurable entry points.
            torch_op = torch.std_mean
        elif variant == "std_mean_correction_names":
            torch_op = torch.std_mean

        super().__init__(
            op_name=variant,
            torch_op=torch_op,
            dtypes=[dtype],
            gems_op=gems_op,
        )
        self.variant = variant
        self.dtype = dtype

    def set_dtypes(self, user_desired_dtypes):
        if user_desired_dtypes and self.dtype not in user_desired_dtypes:
            pytest.skip(f"{self.dtype} was not selected by --dtypes")
        self.to_bench_dtypes = [self.dtype]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if math.prod(shape) <= self.MAX_ELEMENTS
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = _make_input(shape, dtype, self.device)
            dim = [-1]

            if self.variant == "std_mean":
                yield (inp,)
            elif self.variant == "std_mean_dim":
                yield inp, dim, True, False
            elif self.variant == "std_mean_correction":
                yield inp, [], {"correction": 0.5, "keepdim": False}
            elif self.variant == "std_mean_names_dim":
                names = tuple(f"dim_{index}" for index in range(inp.ndim))
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "Named tensors and all their associated APIs are an "
                            "experimental feature.*"
                        ),
                        category=UserWarning,
                    )
                    named_inp = inp.refine_names(*names)
                yield named_inp, [names[-1]], True, False
            elif self.variant == "std_mean_correction_names":
                names = tuple(f"dim_{index}" for index in range(inp.ndim))
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=(
                            "Named tensors and all their associated APIs are an "
                            "experimental feature.*"
                        ),
                        category=UserWarning,
                    )
                    named_inp = inp.refine_names(*names)
                yield named_inp, [names[-1]], {
                    "correction": -0.5,
                    "keepdim": False,
                }
            elif self.variant == "std_mean_correction_out":
                output_shape = shape[:-1]
                std_out = torch.empty(
                    output_shape, dtype=_std_dtype(dtype), device=self.device
                )
                mean_out = torch.empty(output_shape, dtype=dtype, device=self.device)
                yield inp, dim, {
                    "correction": 0.5,
                    "keepdim": False,
                    "out0": std_out,
                    "out1": mean_out,
                }
            else:
                raise ValueError(
                    f"unsupported std_mean benchmark variant: {self.variant}"
                )


@pytest.mark.std_mean
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES["std_mean"])
def test_std_mean(dtype):
    StdMeanBenchmark("std_mean", dtype).run()


@pytest.mark.std_mean_dim
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES["std_mean_dim"])
def test_std_mean_dim(dtype):
    StdMeanBenchmark("std_mean_dim", dtype).run()


@pytest.mark.std_mean_correction
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES["std_mean_correction"])
def test_std_mean_correction(dtype):
    StdMeanBenchmark("std_mean_correction", dtype).run()


@pytest.mark.std_mean_names_dim
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES["std_mean_names_dim"])
def test_std_mean_names_dim(dtype):
    StdMeanBenchmark("std_mean_names_dim", dtype).run()


@pytest.mark.std_mean_correction_names
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES["std_mean_correction_names"])
def test_std_mean_correction_names(dtype):
    StdMeanBenchmark("std_mean_correction_names", dtype).run()


@pytest.mark.std_mean_correction_out
@pytest.mark.parametrize("dtype", STD_MEAN_DTYPES["std_mean_correction_out"])
def test_std_mean_correction_out(dtype):
    StdMeanBenchmark("std_mean_correction_out", dtype).run()
