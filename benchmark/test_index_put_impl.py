import numpy as np
import pytest
import torch

import flag_gems

from . import base, consts


class IndexPutAccFalseBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        INDEX_PUT_SHAPE = (
            ((2**28,), ((2**16,),), (2**16,), False),
            ((32, 32), ((8,), (8,)), (8,), False),
            ((32, 32), ((8,), (2, 8)), (8,), False),
            ((32, 32), ((2, 8),), (32,), False),
            ((1024, 1024), ((64,), (64,)), (64,), False),
            (
                (1024, 1024),
                (
                    (64,),
                    (
                        4,
                        64,
                    ),
                ),
                (64,),
                False,
            ),
            (
                (1024, 1024),
                (
                    (
                        4,
                        64,
                    ),
                ),
                (1024,),
                False,
            ),
            ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
            ((512, 512, 512), ((2, 128), (128,), (128,)), (128,), False),
            ((512, 512, 512), ((2, 128),), (512,), False),
            ((100,), ((100,),), (100,), True),
            ((32, 32), ((32, 32),), (32, 32), True),
            ((16, 16, 4), ((16, 16, 4),), (16, 16, 4), True),
            ((1024, 1024), ((1024, 1024),), (1024 * 1024,), True),
        )
        self.shapes = INDEX_PUT_SHAPE
        return None


def gen_indices_bool(input_shape, indices_shape, accumulate, is_bool):
    indices = []

    if is_bool:
        mask_shape = indices_shape[0]

        mask = torch.randint(
            0, 2, size=mask_shape, dtype=torch.bool, device=flag_gems.device
        )
        return [mask]

    else:
        for i, shape in enumerate(indices_shape):
            index = np.random.choice(
                np.arange(input_shape[i]), size=shape, replace=accumulate
            )
            indices.append(torch.tensor(index, device=flag_gems.device))
        return indices


def _input_fn(accumulate, unsafe=False):
    def inner(shapes, dtype, device):
        input_shape, indices_shape, values_shape, is_bool = shapes
        inp = torch.randn(
            input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

        indices = gen_indices_bool(input_shape, indices_shape, accumulate, is_bool)

        if is_bool:
            K = indices[0].sum().item()
            values = torch.randn(
                (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
            )
        else:
            values = torch.randn(
                values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
            )
        yield inp, indices, values, accumulate, unsafe

    return inner


@pytest.mark.index_put_impl
def test_index_put_impl_acc_false():
    bench = IndexPutAccFalseBenchmark(
        op_name="_index_put_impl_",
        torch_op=torch._index_put_impl_,
        input_fn=_input_fn(False, unsafe=False),
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class IndexPutAccTrueBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        INDEX_PUT_SHAPE = (
            ((2**28,), ((2**16,),), (2**16,), False),
            ((32, 32), ((8,), (8,)), (8,), False),
            ((1024, 1024), ((64,), (64,)), (64,), False),
            ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
            ((512, 512, 512), ((2, 128), (2, 128), (2, 128)), (2, 128), False),
            ((512, 512), ((512, 512),), (512 * 512,), True),
            ((64, 64, 64), ((64, 64, 64),), (64**3,), True),
        )
        self.shapes = INDEX_PUT_SHAPE

        return None


@pytest.mark.index_put_impl
def test_index_put_impl_acc_true():
    bench = IndexPutAccTrueBenchmark(
        op_name="_index_put_impl_",
        torch_op=torch._index_put_impl_,
        input_fn=_input_fn(True, unsafe=False),
        dtypes=[torch.float16, torch.float32],
    )

    bench.run()
