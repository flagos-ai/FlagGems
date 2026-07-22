import numpy as np
import pytest
import torch

import flag_gems

from . import base

# Comprehensive shapes covering all usage patterns
# Format: (input_shape, [index_shapes], values_shape, accumulate)
_SHAPES_ACC_FALSE = (
    # --- 1D input with 1 index ---
    ((256,), ((64,),), (64,), False),
    ((4096,), ((1024,),), (1024,), False),
    ((2**28,), ((2**16,),), (2**16,), False),
    # --- 2D input, 1 index ---
    ((32, 32), ((2, 8),), (32,), False),
    # Replace=False: index elements must be ≤ input dim size
    # so max index elements = input_shape[i]
    ((256, 256), ((16, 16),), (256,), False),  # 256 ≤ 256
    ((1024, 1024), ((64,),), (1024,), False),
    ((1024, 1024), ((4, 64),), (1024,), False),
    ((4096, 4096), ((256,),), (4096,), False),
    # --- 2D input, 2 indices ---
    ((32, 32), ((8,), (8,)), (8,), False),
    ((32, 32), ((8,), (2, 8)), (8,), False),
    ((1024, 1024), ((64,), (64,)), (64,), False),
    ((1024, 1024), ((64,), (4, 64)), (64,), False),
    ((4096, 4096), ((256,), (256,)), (256,), False),
    # --- 3D input (various index configs) ---
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8), False),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
    ((512, 512, 512), ((2, 128), (128,), (128,)), (128,), False),
    ((512, 512, 512), ((2, 128),), (512,), False),
    ((256, 256, 256), ((64,), (64,), (64,)), (64,), False),
    # --- 4D input ---
    ((64, 64, 64, 64), ((8, 8), (8, 8), (8, 8), (8, 8)), (8, 8), False),
)

_SHAPES_ACC_TRUE = (
    # 1D input, 1 index, no suffix
    ((100,), ((100,),), (100,), True),
    ((10000,), ((10000,),), (10000,), True),
    # 2D input, 2 indices (both dims indexed), no suffix
    # values_shape = broadcast_index_shape
    ((32, 32), ((8,), (8,)), (8,), True),
    ((512, 512), ((512,), (512,)), (512,), True),
    ((1024, 1024), ((1024,), (1024,)), (1024,), True),
    # 3D input, 3 indices, no suffix
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8), True),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,), True),
    ((512, 512, 512), ((2, 128), (2, 128), (2, 128)), (2, 128), True),
)


def _gen_indices(input_shape, indices_shapes, accumulate):
    """Generate index tensors for benchmarking."""
    indices = []
    for i, shape in enumerate(indices_shapes):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


def _is_float_dtype(dtype):
    return dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16)


def _make_tensor(shape, dtype, device):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(torch.bool)
    elif _is_float_dtype(dtype):
        return torch.randn(shape, dtype=dtype, device=device)
    else:
        return torch.randint(1, 10, shape, device=device).to(dtype)


def unsafe_index_put_input_fn(shapes, dtype, device):
    """Input generator for _unsafe_index_put benchmark."""
    input_shape, indices_shapes, values_shape, accumulate = shapes
    inp = _make_tensor(input_shape, dtype, device)
    indices = _gen_indices(input_shape, indices_shapes, accumulate)
    values = _make_tensor(values_shape, dtype, device)
    yield inp, indices, values, accumulate


class UnsafeIndexPutBenchmark(base.Benchmark):
    """Dedicated benchmark for _unsafe_index_put with 4-tuple shapes."""

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from unsafe_index_put_input_fn(shape, cur_dtype, self.device)

    def get_gbps(self, args, latency):
        """Compute GB/s: input + output + values + indices."""
        inp = args[0]
        indices = args[1]
        values = args[2]
        io_amount = sum(
            flag_gems.utils.shape_utils.size_in_bytes(t)
            for t in [inp, inp]  # read + write
        )
        io_amount += flag_gems.utils.shape_utils.size_in_bytes(values)
        for idx in indices:
            io_amount += flag_gems.utils.shape_utils.size_in_bytes(idx)
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_shapes(self, shape_file_path=None):
        """Override to use only our shapes, not DEFAULT_SHAPES."""
        # Skip the default shape loading; call set_more_shapes directly
        if hasattr(self, "set_more_shapes") and callable(
            getattr(self, "set_more_shapes")
        ):
            self.shapes = list(self.set_more_shapes() or [])
        if not self.shapes:
            self.shapes = base.Benchmark.DEFAULT_SHAPES


class UnsafeIndexPutAccFalseBenchmark(UnsafeIndexPutBenchmark):
    def set_more_shapes(self):
        return list(_SHAPES_ACC_FALSE)


class UnsafeIndexPutAccTrueBenchmark(UnsafeIndexPutBenchmark):
    def set_more_shapes(self):
        return list(_SHAPES_ACC_TRUE)


BENCH_FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
BENCH_INT_DTYPES = [torch.int32, torch.int64]


@pytest.mark.unsafe_index_put
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_unsafe_index_put_acc_false_floats():
    bench = UnsafeIndexPutAccFalseBenchmark(
        op_name="_unsafe_index_put",
        torch_op=torch._unsafe_index_put,
        input_fn=unsafe_index_put_input_fn,
        dtypes=BENCH_FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unsafe_index_put
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_unsafe_index_put_acc_false_ints():
    bench = UnsafeIndexPutAccFalseBenchmark(
        op_name="_unsafe_index_put",
        torch_op=torch._unsafe_index_put,
        input_fn=unsafe_index_put_input_fn,
        dtypes=BENCH_INT_DTYPES,
    )
    bench.run()


@pytest.mark.unsafe_index_put
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_unsafe_index_put_acc_true_floats():
    bench = UnsafeIndexPutAccTrueBenchmark(
        op_name="_unsafe_index_put",
        torch_op=torch._unsafe_index_put,
        input_fn=unsafe_index_put_input_fn,
        dtypes=BENCH_FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unsafe_index_put
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_unsafe_index_put_acc_true_ints():
    bench = UnsafeIndexPutAccTrueBenchmark(
        op_name="_unsafe_index_put",
        torch_op=torch._unsafe_index_put,
        input_fn=unsafe_index_put_input_fn,
        dtypes=BENCH_INT_DTYPES,
    )
    bench.run()
