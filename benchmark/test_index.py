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
        )
        self.shapes = INDEX_SHAPE
        return None


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
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
def test_index_acc_perf():
    bench = IndexAccBenchmark(
        op_name="index",
        torch_op=torch.ops.aten.index,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.index)

    bench.run()
