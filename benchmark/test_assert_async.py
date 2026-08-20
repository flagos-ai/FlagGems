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


def _input_fn(shape, dtype, device):
    if dtype == torch.bool:
        tensor = torch.ones(shape, dtype=dtype, device=device)
    else:
        tensor = torch.ones(shape, dtype=dtype, device=device)

    msg = "Benchmark assert_async"

    yield (
        tensor,
        msg,
    )


class AssertAsyncBenchmark(base.GenericBenchmark):
    # TODO(Qiming): Is this necessary?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (),
            (1,),
            (1, 1),
            (1, 1, 1),
        ]

    def set_more_shapes(self):
        return None


@pytest.mark.assert_async
def test_assert_async():
    bench = AssertAsyncBenchmark(
        op_name="assert_async",
        input_fn=_input_fn,
        torch_op=torch._assert_async,
        dtypes=[
            torch.bool,
            torch.int32,
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )

    bench.set_gems(flag_gems._assert_async)
    bench.run()
