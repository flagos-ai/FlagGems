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

import csv
import math
import tempfile
from pathlib import Path

import pytest
import torch
import triton
from packaging import version

import flag_gems
from flag_gems import _embedding_bag, _embedding_bag_forward_only

from . import base

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend"
    and version.parse(triton.__version__) < version.parse("3.5"),
    reason="CANN 8.5 is an accuracy-only validation target",
)

SHAPES = [
    # Bags, embedding dimension, table rows, samples per bag.
    (8, 16, 128, 4),
    (32, 64, 1024, 8),
    (128, 128, 4096, 16),
    (256, 256, 4096, 32),
    (1024, 64, 16384, 64),
    (128, 513, 4096, 8),
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


def _npu_complete_kernel_latency(fn, warmup=5, active=30, profile_dir=None):
    # CANN's timer assumes one kernel per call. Native embedding_bag also
    # launches index casts, and Gems may launch a separate validation kernel.
    # Account for the complete, repeated kernel sequence on both sides.
    with tempfile.TemporaryDirectory(prefix="embedding-bag-profile-") as temporary:
        directory = Path(profile_dir) if profile_dir is not None else Path(temporary)
        triton.backends.ascend.testing.do_bench_npu(
            fn, warmup=warmup, active=active, prof_dir=str(directory), keep_res=True
        )
        files = list(directory.rglob("kernel_details.csv"))
        if len(files) != 1:
            raise RuntimeError("Expected one embedding-bag NPU kernel trace")
        with files[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows.sort(key=lambda row: float(row["Start Time(us)"]))
        iterations = warmup + active
        if not rows or len(rows) % iterations:
            raise RuntimeError("NPU trace does not contain complete repeated calls")
        width = len(rows) // iterations
        pattern = [(row["Name"], row["Type"]) for row in rows[:width]]
        durations = []
        for iteration in range(iterations):
            group = rows[iteration * width : (iteration + 1) * width]
            if [(row["Name"], row["Type"]) for row in group] != pattern:
                raise RuntimeError("NPU kernel sequence changed between calls")
            durations.append(math.fsum(float(row["Duration(us)"]) for row in group))
        return math.fsum(durations[warmup:]) / active / 1000


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
        **kwargs,
    ):
        self.bag_mode = mode
        self.weighted = weighted
        self.frequency = frequency
        self.sparse = sparse
        self.backward = backward
        self.forward_only = forward_only
        super().__init__(*args, **kwargs)

    def get_latency(self, op, *args, **kwargs):
        if (
            flag_gems.vendor_name == "ascend"
            and base.Config.mode == base.consts.BenchMode.KERNEL
        ):
            return _npu_complete_kernel_latency(lambda: op(*args, **kwargs))
        return super().get_latency(op, *args, **kwargs)

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
            yield forward_args


@pytest.mark.embedding_bag
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
