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

import numpy as np
import pytest
import torch

import flag_gems

from . import base, consts


class IndexAccBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        INDEX_SHAPE = (
            ((2**28,), ((2**16,),)),
            ((32, 32), ((8,), (8,))),
            ((32, 32), ((8,), (2, 8))),
            ((32, 32), ((2, 8),)),
            ((1024, 1024), ((64,), (64,))),
            ((512, 512, 512), ((128,), (128,), (128,))),
            ((512, 512, 512), ((2, 128), (2, 128), (2, 128))),
            ((512, 512, 512), ((2, 128), (128,), (128,))),
            ((512, 512, 512), ((2, 128),)),
            (
                (64, 64, 64),
                (
                    (2, 8),
                    (2, 8),
                ),
            ),
            # Non-leading adjacent tensor indices. These cover patterns such
            # as x[:, idx, :] and x[:, idx0, idx1, ...] without adding the
            # larger model-scale stress cases to the shared benchmark.
            ((1, 4096, 512), (None, (32768,), None)),
            ((4, 512, 128), (None, (4096,), None)),
            ((2, 256, 256, 64), (None, (4096,), (4096,), None)),
            ((2, 128, 128, 128), (None, (2048,), (2048,), (2048,))),
            (
                (1, 128, 128, 64, 8),
                (None, (2048,), (2048,), (2048,), None),
            ),
        )
        self.shapes = INDEX_SHAPE
        return None


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for dim, shape in enumerate(indices_shape):
        if shape is None:
            indices.append(None)
            continue
        index = np.random.choice(
            np.arange(input_shape[dim]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))

    return indices


def _input_fn(shapes, dtype, device):
    input_shape, indices_shape = shapes
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, True)

    yield inp, indices


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_index_acc_perf():
    bench = IndexAccBenchmark(
        op_name="index",
        torch_op=torch.ops.aten.index,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.index)

    bench.run()
