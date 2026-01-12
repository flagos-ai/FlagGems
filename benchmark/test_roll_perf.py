from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input


class RollBenchmark(Benchmark):
    """
    Benchmark for single-dimension roll operation.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["gbps"]

    def set_more_shapes(self):
        # 1D: various sizes
        # 2D: square and rectangular
        # 3D: cube-like
        # 4D: large batched tensors
        return [
            (1024,),
            (1024, 1024),
            (64, 64, 64),
            (16, 128, 128, 128),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            shift = shape[0] // 3 if len(shape) > 0 else 1
            dim = 0
            yield inp, shift, dim

    def get_gbps(self, op, *args, **kwargs):
        inp = op[0]
        latency = kwargs.get("latency")
        numel = inp.numel()
        element_size = inp.element_size()
        gb = 2 * numel * element_size / 1e9
        return gb / (latency * 1e-3)


@pytest.mark.roll
def test_perf_roll():
    def torch_op(inp, shift, dim):
        return torch.roll(inp, shift, dims=dim)

    bench = RollBenchmark(
        op_name="roll",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class RollMultiDimBenchmark(Benchmark):
    """
    Benchmark for multi-dimension roll operation.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["gbps"]

    def set_more_shapes(self):
        return [
            (64, 64, 64),
            (128, 256, 256),
            (16, 128, 128, 128),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            rank = len(shape)
            if rank >= 2:
                shifts = [shape[0] // 3, shape[1] // 4]
                dims = [0, 1]
            else:
                shifts = [shape[0] // 3]
                dims = [0]
            yield inp, shifts, dims

    def get_gbps(self, op, *args, **kwargs):
        inp = op[0]
        latency = kwargs.get("latency")
        numel = inp.numel()
        element_size = inp.element_size()
        gb = 2 * numel * element_size / 1e9
        return gb / (latency * 1e-3)


@pytest.mark.roll
def test_perf_roll_multi_dim():
    def torch_op(inp, shifts, dims):
        return torch.roll(inp, shifts, dims=dims)

    bench = RollMultiDimBenchmark(
        op_name="roll_multi",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class RollFlattenBenchmark(Benchmark):
    """
    Benchmark for flattened roll (dims=None).
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["gbps"]

    def set_more_shapes(self):
        return [
            (1024, 1024),
            (64, 64, 64),
            (128, 256, 256),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            shift = inp.numel() // 3
            yield inp, shift

    def get_gbps(self, op, *args, **kwargs):
        inp = op[0]
        latency = kwargs.get("latency")
        numel = inp.numel()
        element_size = inp.element_size()
        gb = 2 * numel * element_size / 1e9
        return gb / (latency * 1e-3)


@pytest.mark.roll
def test_perf_roll_flatten():
    def torch_op(inp, shift):
        return torch.roll(inp, shift, dims=None)

    bench = RollFlattenBenchmark(
        op_name="roll_flatten",
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
