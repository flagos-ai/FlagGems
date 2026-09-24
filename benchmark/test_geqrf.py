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

from . import base, generated_operator_utils

# Householder QR shapes (*batch, M, N). core_shapes.yaml has no geqrf entry, so
# these are the built-in defaults for both levels; a caller-supplied shape file
# naming `geqrf` or `GeqrfBenchmark` still overrides them (see GeqrfBenchmark).
GEQRF_BENCH_SHAPES = [
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2048, 1024),
    (1024, 2048),
    (128, 32),
    (512, 64),
    (32, 128),
    (64, 256),
    (64, 8, 8),
    (4, 1024, 1024),
]

# Native geqrf implements float32/float64/complex64/complex128 only: the fp16 and
# bfloat16 entries of the shared float dtype list fail with geqrf_cuda not
# implemented for 'Half'/'BFloat16', and the two wide types need backend fp64
# support.
GEQRF_BENCH_DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    GEQRF_BENCH_DTYPES += [torch.float64, torch.complex128]


def _check_case_shape(dims):
    # Metadata-only validation of the public case record: geqrf needs rank >= 2
    # and every dimension must be a non-negative integer. No tensor is created or
    # inspected here, so --list-cases stays allocation free.
    if len(dims) < 2:
        raise ValueError(f"geqrf requires an input of rank >= 2, got {dims}")
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
            raise ValueError(f"geqrf requires non-negative integer dims, got {dims}")
    return dims


def _case_fn(shape, dtype):
    # Listing builds metadata only: shape and params stay JSON compatible while
    # the torch objects live in builder_args.
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": _check_case_shape(list(shape))},
        params={},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return (torch.randn(shape, dtype=dtype, device=device),)


class GeqrfBenchmark(generated_operator_utils.OperatorBenchmark):
    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        # OperatorBenchmark.set_shapes uses a `geqrf`/`GeqrfBenchmark` shape-file
        # entry and else the supplied default_shapes; without this wrapper the
        # MRO fallback of GenericBenchmark.set_shapes would match the generic
        # `Benchmark:` entry of core_shapes.yaml, whose rank-1 and cubic shapes
        # are not valid geqrf inputs. Passing default_shapes keeps the QR shapes
        # above authoritative while a custom shape file still wins.
        # set_more_shapes is deliberately not overridden: OperatorBenchmark.
        # set_shapes never merges it, so the listed cases are identical at the
        # core and comprehensive levels.
        if default_shapes is None:
            default_shapes = GEQRF_BENCH_SHAPES
        return super().set_shapes(shape_file_path, default_shapes=default_shapes)


@pytest.mark.geqrf
def test_geqrf():
    bench = GeqrfBenchmark(
        op_name="geqrf",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.geqrf,
        gems_op=getattr(flag_gems, "geqrf", None),
        dtypes=GEQRF_BENCH_DTYPES,
    )
    bench.run()
