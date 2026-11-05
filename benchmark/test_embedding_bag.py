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
from flag_gems import _embedding_bag, _embedding_bag_forward_only

from . import base

SHAPES = [
    # Bags, embedding dimension, table rows, samples per bag.
    (8, 16, 128, 4),
    (32, 64, 1024, 8),
    (128, 128, 4096, 16),
    (256, 256, 4096, 32),
    (1024, 64, 16384, 64),
    (128, 512, 4096, 8),
]
DTYPES = [
    torch.float16,
    torch.float32,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_bf16,
            reason="Device does not support bfloat16",
        ),
    ),
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            not flag_gems.runtime.device.support_fp64,
            reason="Device does not support float64",
        ),
    ),
]
FORWARD_MODES = [(0, False), (1, False), (2, False), (0, True)]


class EmbeddingBagBenchmark(base.Benchmark):
    def __init__(
        self,
        *args,
        mode=0,
        weighted=False,
        frequency=False,
        sparse=False,
        backward=False,
        forward_only=False,
        default_entry=False,
        **kwargs,
    ):
        self.bag_mode = mode
        self.weighted = weighted
        self.frequency = frequency
        self.sparse = sparse
        self.backward = backward
        self.forward_only = forward_only
        self.default_entry = default_entry
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "num_bags, embedding_dim, num_weights, samples_per_bag"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, cur_dtype):
        for case, (num_bags, dim, num_weights, bag_length) in enumerate(self.shapes):
            num_indices = num_bags * bag_length
            index_dtype = torch.int32 if case % 2 else torch.int64
            if (
                flag_gems.vendor_name == "mthreads"
                and not self.backward
                and not self.forward_only
            ):
                # MUSA 5.2's native _embedding_bag requires int64 helpers. Both
                # implementations receive identical supported benchmark inputs;
                # accuracy tests separately cover the Gems int32 path.
                index_dtype = torch.int64
            include_last = bool(case % 2)
            weight = torch.randn(
                (num_weights, dim), dtype=cur_dtype, device=self.device
            )
            indices = torch.randint(
                0, num_weights, (num_indices,), dtype=index_dtype, device=self.device
            )
            offsets = torch.arange(
                0,
                num_indices + int(include_last),
                bag_length,
                dtype=index_dtype,
                device=self.device,
            )
            psw = (
                torch.rand((num_indices,), dtype=cur_dtype, device=self.device)
                if self.weighted
                else None
            )
            forward_args = (
                weight,
                indices,
                offsets,
                self.frequency,
                self.bag_mode,
                self.sparse,
                psw,
                include_last,
                0,
            )
            yield forward_args[:-1] if self.default_entry else forward_args


@pytest.mark.underscore_embedding_bag
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode,weighted", FORWARD_MODES)
def test_embedding_bag(dtype, mode, weighted):
    bench = EmbeddingBagBenchmark(
        op_name=f"embedding_bag_mode{mode}{'_weighted' if weighted else ''}",
        torch_op=torch.ops.aten._embedding_bag.default,
        gems_op=_embedding_bag,
        dtypes=[dtype],
        mode=mode,
        weighted=weighted,
    )
    bench.run()


@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode,weighted", FORWARD_MODES)
def test_embedding_bag_forward_only(dtype, mode, weighted):
    bench = EmbeddingBagBenchmark(
        op_name=f"embedding_bag_forward_only_mode{mode}{'_weighted' if weighted else ''}",
        torch_op=torch.ops.aten._embedding_bag_forward_only.default,
        gems_op=_embedding_bag_forward_only,
        forward_only=True,
        dtypes=[dtype],
        mode=mode,
        weighted=weighted,
    )
    bench.run()


@pytest.mark.embedding_bag
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode,weighted", FORWARD_MODES)
def test_embedding_bag_default(dtype, mode, weighted):
    bench = EmbeddingBagBenchmark(
        op_name=f"embedding_bag_default_mode{mode}{'_weighted' if weighted else ''}",
        torch_op=torch.ops.aten.embedding_bag.default,
        gems_op=flag_gems.embedding_bag,
        default_entry=True,
        dtypes=[dtype],
        mode=mode,
        weighted=weighted,
    )
    bench.run()
