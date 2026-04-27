from typing import Generator

import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class NLLLossNDBenchmark(utils.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes = [
            (64, 64),
            (256, 256),
            (10000, 65536),
            (32, 128, 512),
            (64, 64, 4, 8),
            (256, 256, 4, 8),
            (4096, 4096, 4, 8),
            (64, 64, 8, 4, 8),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


def nll_loss_nd_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    inp = torch.nn.functional.log_softmax(inp, dim=1)

    target_shape = list(shape)
    del target_shape[1]
    C = shape[1]
    target = torch.randint(0, C, target_shape, dtype=torch.long, device=device)

    yield inp, target

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        weight_tensor = torch.rand(C, dtype=cur_dtype, device=device)
        for weight in [weight_tensor, None]:
            for reduction in ["none", "mean", "sum"]:
                yield inp, target, {
                    "weight": weight,
                    "ignore_index": 1,
                    "reduction": reduction,
                }


@pytest.mark.nll_loss_nd
def test_nll_loss_nd():
    bench = NLLLossNDBenchmark(
        input_fn=nll_loss_nd_input_fn,
        op_name="nll_loss_nd",
        torch_op=torch.nn.functional.nll_loss,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
