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

from . import base, consts


def _input_fn(shape, dtype, device):
    dist = torch.rand(shape, dtype=dtype, device=device)
    n_categories = shape[-1] if len(shape) > 1 else shape[0]
    batch = shape[0] if len(shape) > 1 else 1

    # NPU's native torch.multinomial (baseline) allocates an internal tensor
    # roughly proportional to  batch * n_samples * n_categories * sizeof(float)
    # with a large constant factor (~8x observed).  Keep the product under a
    # safe budget to avoid OOM on a 60 GiB NPU.
    MEM_BUDGET = 8 * (1024**3)  # 8 GiB budget (conservative)
    BYTES_PER_ELEM = 4 * 8  # float32 * ~8x overhead factor
    max_samples = MEM_BUDGET // max(batch * n_categories * BYTES_PER_ELEM, 1)
    n_samples = min(n_categories, max_samples)
    n_samples = max(n_samples, 1)  # at least 1 sample

    yield dist, n_samples, True,


@pytest.mark.multinomial
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_multinomial_with_replacement():
    # Custom shapes for multinomial to avoid OOM on NPU and stay within
    # the n_categories <= 2^24 hardware limit of aclnnMultinomial.
    # The generic Benchmark shapes from core_shapes.yaml include
    # 1-D / 3-D shapes and entries > 2^24 that are unsuitable.
    _MULTINOMIAL_SHAPES = [
        (1, 1000),  # small single distribution
        (1, 10000),  # medium single distribution
        (100, 1000),  # small batch
        (1000, 1000),  # balanced batch
        (10000, 256),  # large batch, narrow categories
        (1000, 4096),  # medium batch, wide categories
    ]

    class MultinomialBenchmark(base.GenericBenchmark):
        DEFAULT_SHAPE_DESC = "batch_size, n_categories"

        def set_shapes(self, shape_file_path=None):
            # Bypass yaml / DEFAULT_SHAPES entirely to ensure only
            # multinomial-safe shapes are used.
            self.shapes = list(_MULTINOMIAL_SHAPES)

        def set_more_shapes(self):
            return []

        def get_input_iter(self, dtype):
            for shape in self.shapes:
                yield from self.input_fn(shape, dtype, self.device)

    bench = MultinomialBenchmark(
        input_fn=_input_fn,
        op_name="multinomial",
        torch_op=torch.multinomial,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
