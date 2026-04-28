import pytest
import torch

from . import base, consts


class TopKBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (4096, 4096),
            (10000, 256),
            (10000, 65536),
            (4, 128),
            (8, 256),
            (64, 128, 8),
            (64, 1024, 32),
            (64, 8192, 128),
            (128, 32768, 256),
            ((4, 128, 64), 5),
            ((4, 128, 64), 64),
            ((8, 512, 32), 32),
            ((16, 1024, 256), 256),
        ]


def _input_fn(shape, dtype, device):
    if len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        x_shape, k = shape
        x = torch.randn(x_shape, device=device, dtype=dtype)
        yield {"x": x, "k": k, "dim": -1},
    elif len(shape) == 3:
        m, n, k = shape
        x = torch.randn((m, n), device=device, dtype=dtype)
        yield {"x": x, "k": k, "dim": -1},
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
        k = 5 if shape[-1] > 5 else shape[-1]
        yield {"x": x, "k": k, "dim": -1},
    # TODO:  Currently only support sorted == True and only support topk in last dimension
    # if Config.bench_level == BenchLevel.COMPREHENSIVE:
    #     k = 5 if shape[0] > 5 else shape[0]
    #     yield {"x": x, "k": k, "dim": 0},
    #     yield {"x": x, "k": k, "dim": -1, "sorted": False},


@pytest.mark.topk
def test_topk():
    bench = TopKBenchmark(
        op_name="topk",
        input_fn=_input_fn,
        torch_op=torch.topk,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
