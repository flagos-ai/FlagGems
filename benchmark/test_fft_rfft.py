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

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None)
# Real-signal lanes only, measured with torch.ops.aten.fft_rfft on this
# backend: float32 -> complex64, float64 -> complex128, float16 -> complex32.
# bfloat16 and the fp8 dtypes raise 'Unsupported dtype ...' at the native call,
# so they are not benchmarked. float16 is a cuFFT capability whose transform
# additionally needs a power-of-two signal length (cuFFT reports the
# restriction for the transformed axis only), so that lane plans power-of-two
# transform lengths and drops the odd / non-power-of-two rows.
_HALF_FFT_SUPPORTED = flag_gems.vendor_name == "nvidia"
_FP64_SUPPORTED = flag_gems.runtime.device.support_fp64

_RFFT_DTYPES = [torch.float32]
if _HALF_FFT_SUPPORTED:
    _RFFT_DTYPES.append(torch.float16)
if _FP64_SUPPORTED:
    _RFFT_DTYPES.append(torch.float64)

# Performance-relevant real signals, 1-D to 3-D and both signal-heavy and
# batch-heavy. The stock default shape list does not describe FFT workloads, so
# these shapes are used instead; an fft_rfft section of a caller-supplied shape
# file still takes precedence (OperatorBenchmark.set_shapes).
_RFFT_SHAPES = [
    (256,),
    (1024,),
    (1024, 1024),
    (2048, 2048),
    (64, 1024),
    (8, 512, 512),
    (16, 128, 256),
]


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _call_kwargs(n, dim, norm):
    # The single source of truth for the call: params and the executed call are
    # both derived from it, so listing and replay cannot drift apart.
    kwargs = {}
    if n is not None:
        kwargs["n"] = n
    if dim != -1:
        kwargs["dim"] = dim
    if norm is not None:
        kwargs["norm"] = norm
    return kwargs


def _call_params(n, dim, norm):
    return {
        "n": "default" if n is None else n,
        "dim": "default" if dim == -1 else dim,
        "norm": "default" if norm is None else norm,
    }


def _validate_shape(shape):
    # Static validation of a shape descriptor (default list or shape file):
    # rfft transforms one axis of a rank >= 1 tensor, so a 0-dim input
    # (IndexError natively) and any zero-length axis (RuntimeError natively)
    # are rejected while planning instead of being listed.
    sizes = tuple(shape) if isinstance(shape, (tuple, list)) else ()
    if len(sizes) < 1:
        raise ValueError(
            "fft_rfft needs a descriptor of rank >= 1; a 0-dim input is rejected."
        )
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"fft_rfft shape entries must be integers, got {size!r}.")
        if size < 1:
            raise ValueError(
                f"fft_rfft cannot transform a zero-length axis, got {sizes}."
            )
    return sizes


def _validate_call(sizes, n, dim, norm):
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        raise ValueError(f"fft_rfft requires an integer n >= 1, got {n!r}.")
    if (
        isinstance(dim, bool)
        or not isinstance(dim, int)
        or not -len(sizes) <= dim < len(sizes)
    ):
        raise ValueError(f"fft_rfft dim {dim!r} is out of range for rank {len(sizes)}.")
    if norm not in (None, "backward", "forward", "ortho"):
        raise ValueError(
            f"fft_rfft norm must be None/backward/forward/ortho, got {norm!r}."
        )


def _plan_rows(shape):
    # (n, dim, norm) request per family: the natural length, a truncating n, a
    # padding n, an odd n + 1, and a leading-axis transform.
    natural = shape[-1]
    return [
        (None, -1, None),
        (max(1, natural // 2), -1, "ortho"),
        (_next_power_of_two(natural), -1, "forward"),
        (natural + 1, -1, None),
        (4, 0, "ortho"),
    ]


def _transform_length(sizes, n, dim):
    # The axis rfft actually transforms: with n omitted it is the axis selected
    # by dim, which is the leading axis for the dim=0 family.
    return sizes[dim] if n is None else n


def _case_fn(shape, dtype):
    sizes = _validate_shape(shape)
    for n, dim, norm in _plan_rows(sizes):
        _validate_call(sizes, n, dim, norm)
        if dtype is torch.float16 and not _is_power_of_two(
            _transform_length(sizes, n, dim)
        ):
            continue
        yield base.BenchmarkCasePlan(
            shape={"input": sizes},
            params=_call_params(n, dim, norm),
            builder_args=(sizes, n, dim, norm),
        )


def _make_input(shape, dtype, device):
    # The benchmarked lanes are the real floating dtypes, and every real
    # floating dtype has a direct random builder, so no shared generator
    # fallback is needed.
    return torch.randn(shape, dtype=dtype, device=device)


def _build_inputs_fn(plan, dtype, device):
    shape, n, dim, norm = plan.builder_args
    inp = _make_input(shape, dtype, device)
    return inp, _call_kwargs(n, dim, norm)


class FftRfftBenchmark(OperatorBenchmark):
    # set_shapes feeds both --list-cases and execution; a caller-supplied shape
    # file that defines fft_rfft still wins.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_RFFT_SHAPES)


@pytest.mark.fft_rfft
def test_fft_rfft():
    bench = FftRfftBenchmark(
        op_name="fft_rfft",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fft_rfft,
        gems_op=getattr(flag_gems, "fft_rfft", None),
        dtypes=_RFFT_DTYPES,
    )
    bench.run()
