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

import flag_gems
import pytest
import torch

from . import base, consts


_HYGON_COMMON_ROWS = [
    1,
    2,
    4,
    8,
    16,
    *range(24, 257, 8),
    *range(272, 513, 16),
]

HYGON_MV_REAL_MODEL_SHAPES = [
    *[(n, 2048) for n in _HYGON_COMMON_ROWS],
    (1034, 2048),
    (1035, 2048),
    (1036, 2048),
    (1352, 2048),
    (2048, 2048),
    (3104, 2048),
    (4107, 2048),
    (4108, 2048),
    (4138, 2048),
    (4434, 2048),
    (4435, 2048),
    (4436, 2048),
    (9309, 2048),
    (9330, 2048),
    (11469, 2048),
    (13421, 2048),
    (13422, 2048),
    (16384, 2048),
    *[(n, 4096) for n in _HYGON_COMMON_ROWS],
    (1035, 4096),
    (1036, 4096),
    (2048, 4096),
    (4107, 4096),
    (4108, 4096),
    (4434, 4096),
    (4435, 4096),
    (5240, 4096),
    (8214, 4096),
    (13421, 4096),
    (13422, 4096),
    (16384, 4096),
]

HYGON_MV_REPRESENTATIVE_SHAPES = [
    (1, 2048),
    (4, 2048),
    (8, 2048),
    (16, 2048),
    (32, 2048),
    (64, 2048),
    (128, 2048),
    (256, 2048),
    (512, 2048),
    (1035, 2048),
    (1352, 2048),
    (2048, 2048),
    (3104, 2048),
    (4107, 2048),
    (4434, 2048),
    (9309, 2048),
    (11469, 2048),
    (13421, 2048),
    (16384, 2048),
    (1, 4096),
    (4, 4096),
    (8, 4096),
    (16, 4096),
    (32, 4096),
    (64, 4096),
    (128, 4096),
    (256, 4096),
    (512, 4096),
    (1035, 4096),
    (2048, 4096),
    (4107, 4096),
    (4434, 4096),
    (5240, 4096),
    (8214, 4096),
    (13421, 4096),
    (16384, 4096),
]

assert len(HYGON_MV_REAL_MODEL_SHAPES) == 132
assert len(HYGON_MV_REPRESENTATIVE_SHAPES) == 36


class HygonMvBenchmark(base.GenericBenchmark2DOnly):
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
        del shape_file_path
        if base.Config.bench_level == consts.BenchLevel.CORE:
            self.shapes = list(HYGON_MV_REPRESENTATIVE_SHAPES)
        else:
            self.shapes = list(HYGON_MV_REAL_MODEL_SHAPES)
        self.shape_desc = "N, K"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.mv
def test_hygon_mv():
    if flag_gems.vendor_name != "hygon":
        pytest.skip("Hygon MV benchmark requires the Hygon backend")

    bench = HygonMvBenchmark(
        op_name="mv",
        input_fn=_input_fn,
        torch_op=torch.Tensor.mv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
