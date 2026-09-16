# Copyright 2026, The FlagOS Contributors.
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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base

BENCH_DTYPES = [  # The Ascend performance report is scoped to FP16 and BF16.
    torch.float16,
    torch.bfloat16,
]

BENCH_CASES = [
    ((1024,), 0.0, 128, -1),
    ((1024,), 0.1, 128, -1),
    ((16384,), 0.001, 128, -1),
    ((16384,), 0.1, 1024, -1),
    ((262144,), 0.001, 1024, -1),
    ((262144,), 0.1, 4096, -1),
    ((1048576,), 0.001, 1024, -1),
    ((1048576,), 0.1, 4096, -1),
    ((32, 1024), 0.1, 1024, -1),
    ((128, 4096), 0.01, 4096, -1),
    ((1024, 4096), 0.001, 1024, -1),
    ((1024, 4096), 0.1, 4096, -1),
    ((16, 64, 64), 0.1, 1024, -1),
    ((32, 128, 128), 0.01, 4096, -1),
]


def _composed_nonzero_static_baseline(input, size, fill_value=-1):
    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    out = torch.full(
        (size, input.dim()),
        fill_value,
        dtype=torch.int64,
        device=input.device,
    )
    if size == 0 or input.dim() == 0:
        return out

    indices = torch.nonzero(input)
    copy_len = min(size, indices.shape[0])
    if copy_len > 0:
        out[:copy_len].copy_(indices[:copy_len])
    return out


def _make_input(shape, dtype, nnz_ratio, device):
    torch.manual_seed(0)
    mask = torch.rand(shape, device=device) < nnz_ratio

    if dtype == torch.bool:
        return mask

    x = torch.zeros(shape, device=device, dtype=dtype)
    if dtype.is_floating_point:
        x[mask] = 1.0
    else:
        x[mask] = 1
    return x


def _get_baseline_nonzero_static():
    # Nonzero_static is not supported for cuda <= 11.4
    if flag_gems.vendor_name in ("ascend", "hygon", "iluvatar"):
        return _composed_nonzero_static_baseline
    return torch.nonzero_static


def _input_fn(case, dtype, device):
    shape, nnz_ratio, size, fill_value = case
    inp = _make_input(shape, dtype, nnz_ratio, device)
    yield inp, {"size": size, "fill_value": fill_value}


class NonzeroStaticBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = BENCH_CASES
    DEFAULT_SHAPE_DESC = "shape, nnz_ratio, size, fill_value"

    def init_default_config(self):
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)
        # Each case includes shape, sparsity, size, and fill_value, so generic
        # shape-only configuration files are not compatible with this benchmark.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for case in self.shapes:
            yield from self.input_fn(case, dtype, self.device)


@pytest.mark.nonzero_static
def test_perf_nonzero_static():
    baseline_nonzero_static = _get_baseline_nonzero_static()
    bench = NonzeroStaticBenchmark(
        op_name="nonzero_static",
        torch_op=baseline_nonzero_static,
        gems_op=flag_gems.nonzero_static,
        input_fn=_input_fn,
        dtypes=BENCH_DTYPES,
    )
    bench.run()
