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

from . import base, consts
from .conftest import Config

# Each entry is an (N, C) pair. Cost is driven by C rather than by the element
# count, since every active target pairs against every non-target class; the
# pairs hold N * C roughly fixed while sweeping C across the 64-wide class tile
# and over the single-launch cutoff for reduction 0, so both the tall/narrow and
# the few-row/wide ends of the grid are measured.
CORE_SHAPES = [
    (8, 128),
    (32, 256),
    (16, 512),
    (8, 1024),
    (2, 2048),
    (1, 4096),
]

# Wide-and-shallow production shapes, plus a class count that leaves a partial
# class tile so every program takes the masked path.
COMPREHENSIVE_SHAPES = [
    (6400, 15),
    (1024, 1024),
    (32, 257),
]

# The op accepts fp64 and uses a fp64 accumulator for it, which is a separate
# kernel path worth measuring, so it is appended where the device supports it.
_FP64 = (
    # fp64 is not in consts.FLOAT_DTYPES; it is an extra path, not a substitute.
    [torch.float64]
    if flag_gems.runtime.device.support_fp64
    else []
)
_DTYPES = consts.FLOAT_DTYPES + _FP64


class MultilabelMarginLossBenchmark(base.Benchmark):
    def __init__(self, *, reduction, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def set_shapes(self, shape_file_path=None):
        # A loss over an (N, C) input and a same-shaped int64 target needs
        # loss-specific shape pairs, so they are listed here instead of coming
        # from the shared shape file.
        self.shapes = list(CORE_SHAPES)
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_SHAPES)
        self.shape_desc = "N, C"

    def get_input_iter(self, cur_dtype) -> Generator:
        for n, c in self.shapes:
            inp = torch.randn((n, c), dtype=cur_dtype, device=self.device)
            # Half of each row's class ids are active, the rest of the row is
            # the -1 sentinel and the ignored payload behind it.
            target = torch.full((n, c), -1, dtype=torch.int64, device=self.device)
            keep = max(1, c // 2)
            ids = torch.arange(c, dtype=torch.int64, device=self.device)
            target[:, :keep] = ids[:keep]
            yield inp, target, self.reduction


@pytest.mark.multilabel_margin_loss
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss(reduction):
    bench = MultilabelMarginLossBenchmark(
        reduction=reduction,
        op_name="multilabel_margin_loss",
        torch_op=torch.ops.aten.multilabel_margin_loss,
        gems_op=flag_gems.multilabel_margin_loss,
        dtypes=_DTYPES,
    )
    bench.run()
