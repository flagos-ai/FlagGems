import os
from functools import lru_cache
from typing import Generator

import pytest
import torch

import flag_gems

from . import base

BENCH_DTYPES = [torch.float16, torch.bfloat16]  # target report uses fp16 and bf16
ASCEND_BASELINE_SOURCE = r"""
#include <ATen/Functions.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>

namespace {

at::Tensor nonzero_static(
    const at::Tensor& input,
    int64_t size,
    int64_t fill_value) {
  TORCH_CHECK(size >= 0, "nonzero_static: size must be non-negative");

  auto out = at::full(
      {size, input.dim()}, fill_value, input.options().dtype(at::kLong));
  if (size == 0 || input.dim() == 0) {
    return out;
  }

  auto indices = at::nonzero(input);
  auto copy_len = std::min<int64_t>(size, indices.size(0));
  if (copy_len > 0) {
    out.narrow(0, 0, copy_len)
        .copy_(indices.narrow(0, 0, copy_len));
  }
  return out;
}

}

TORCH_LIBRARY(flag_gems_ascend_baseline, m) {
  m.def(
      "nonzero_static(Tensor input, int size, int fill_value=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(
    flag_gems_ascend_baseline,
    CompositeExplicitAutograd,
    m) {
  m.impl("nonzero_static", TORCH_FN(nonzero_static));
}
"""
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


@lru_cache(maxsize=1)
def _load_ascend_baseline():
    from torch.utils.cpp_extension import load_inline

    load_inline(
        name="flag_gems_ascend_nonzero_static_baseline",
        cpp_sources=ASCEND_BASELINE_SOURCE,
        extra_cflags=["-O3"],
        is_python_module=False,
        verbose=os.getenv("FLAGGEMS_ASCEND_BASELINE_VERBOSE", "0") == "1",
    )
    return torch.ops.flag_gems_ascend_baseline.nonzero_static


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
    if flag_gems.vendor_name == "ascend":
        return _load_ascend_baseline()
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
