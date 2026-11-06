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

from contextlib import contextmanager
from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts
from .gemm_shapes import gemm_layouts


class MvBenchmark(base.GenericBenchmark2DOnly):
    @contextmanager
    def _cache_flush_context(self):
        """Keep Hygon MV's cache flush on native fill kernels."""

        if flag_gems.vendor_name != "hygon":
            yield
            return

        original_use_gems = flag_gems.use_gems

        def use_gems_for_mv(*args, **kwargs):
            if kwargs.get("exclude") == ["zero_"]:
                kwargs = dict(kwargs)
                kwargs["exclude"] = [
                    "zero_",
                    "fill_scalar_",
                    "fill_tensor_",
                ]
            return original_use_gems(*args, **kwargs)

        flag_gems.use_gems = use_gems_for_mv
        try:
            yield
        finally:
            flag_gems.use_gems = original_use_gems

    def run(self):
        with self._cache_flush_context():
            return super().run()

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path or self.DEFAULT_SHAPE_FILES)
        self.shape_desc = "N, K"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for n, k in self.shapes:
            for column_major in gemm_layouts("mv", (n, k), base.Config.mm_layout):
                yield from self.input_fn(n, k, dtype, self.device, column_major)

    def get_case_iter(self, dtype):
        ordinal = 0
        for n, k in self.shapes:
            for column_major in gemm_layouts("mv", (n, k), base.Config.mm_layout):
                yield consts.BenchmarkCaseSpec(
                    case_id=self._case_id(dtype, ordinal),
                    ordinal=ordinal,
                    dtype=dtype,
                    shape={"n": n, "k": k},
                    params={"column_major": column_major},
                    builder_args=(n, k, column_major),
                )
                ordinal += 1

    def build_inputs(self, case):
        n, k, column_major = case.builder_args
        return next(self.input_fn(n, k, case.dtype, self.device, column_major))


def _input_fn(n, k, cur_dtype, device, column_major=False):
    inp1 = (
        torch.randn([k, n], dtype=cur_dtype, device=device).t()
        if column_major
        else torch.randn([n, k], dtype=cur_dtype, device=device)
    )
    inp2 = torch.randn([k], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.mv
def test_mv():
    bench = MvBenchmark(
        op_name="mv",
        input_fn=_input_fn,
        torch_op=torch.Tensor.mv,
        dtypes=consts.GEMM_DTYPES,
    )
    bench.run()
