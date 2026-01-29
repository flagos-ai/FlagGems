"""Performance benchmarks for scatter_reduce operator."""
from typing import Generator

import torch

import flag_gems

from .performance_utils import FLOAT_DTYPES, GenericBenchmark


def scatter_reduce_input_fn(shape, dtype, device, reduce="sum"):
    """Generate inputs for scatter_reduce benchmark.

    Yields: inp, dim, index, src, {"reduce": reduce, "include_self": True}

    This matches the signature:
    scatter_reduce(input, dim, index, src, reduce, *, include_self=True)

    Note: 'reduce' is passed as a kwarg because it's a string parameter.
    """
    dim = 0
    index_shape = list(shape)
    index_shape[dim] = max(1, shape[dim] // 4)

    inp = torch.randn(shape, dtype=dtype, device=device)
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device=device)
    src = torch.randn(index_shape, dtype=dtype, device=device)

    # Yield: tensor args, then kwargs dict
    # Args: (input, dim, index, src)
    # Kwargs: {"reduce": reduce, "include_self": True}
    yield inp, dim, index, src, {"reduce": reduce, "include_self": True}


class ScatterReduceBenchmark(GenericBenchmark):
    """Benchmark for scatter_reduce operator."""

    def __init__(self, reduce="sum"):
        """Initialize benchmark with specific reduction mode."""
        self.reduce = reduce
        super().__init__(
            input_fn=lambda shape, dtype, device: scatter_reduce_input_fn(
                shape, dtype, device, reduce=self.reduce
            ),
            op_name=f"scatter_reduce_{reduce}",
            torch_op=torch.scatter_reduce,
            dtypes=FLOAT_DTYPES,
        )

    def get_input_iter(self, cur_dtype) -> Generator:
        """Generate input shapes for benchmark."""
        shapes = [
            # Small sizes
            (100,),
            (64, 64),
            # Medium sizes
            (256, 256),
            (512, 512),
            # Large sizes
            (1024, 1024),
            (2048, 2048),
            # 3D tensors
            (32, 32, 32),
            (64, 64, 64),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


def test_scatter_reduce_sum_perf():
    """Test scatter_reduce sum performance."""
    bench = ScatterReduceBenchmark(reduce="sum")
    bench.set_gems(flag_gems.scatter_reduce)
    bench.run()


def test_scatter_reduce_mean_perf():
    """Test scatter_reduce mean performance."""
    bench = ScatterReduceBenchmark(reduce="mean")
    bench.set_gems(flag_gems.scatter_reduce)
    bench.run()


def test_scatter_reduce_amax_perf():
    """Test scatter_reduce amax performance."""
    bench = ScatterReduceBenchmark(reduce="amax")
    bench.set_gems(flag_gems.scatter_reduce)
    bench.run()


if __name__ == "__main__":
    test_scatter_reduce_sum_perf()
    test_scatter_reduce_mean_perf()
    test_scatter_reduce_amax_perf()
