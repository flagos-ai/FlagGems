from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts

MESHGRID_COMMON_CASES = [
    [16, 16],
    [1024, 1024],
    [16, 16, 16],
    [256, 256, 256],
    [32, 32, 32, 32],
    [128, 128, 128, 128],
]


def _generate_tensors(shapes, dtype, device):
    return [
        torch.linspace(0, size, size, device=device, dtype=dtype) for size in shapes
    ]


class MeshgridBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(MESHGRID_COMMON_CASES)
        if flag_gems.vendor_name == "ascend":
            self.shapes.insert(0, [256, 256, 16])

    def get_input_iter(self, cur_dtype) -> Generator:
        for shapes in self.shapes:
            tensors = _generate_tensors(shapes, cur_dtype, self.device)
            yield tensors, {"indexing": "ij"}

    @staticmethod
    def _torch_meshgrid_with_copy(*args, **kwargs):
        grid_out = torch.meshgrid(*args, **kwargs)
        return [tensor.clone() for tensor in grid_out]

    def measure(self, op, *args, **kwargs):
        import time

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif flag_gems.vendor_name == "ascend":
            torch.npu.empty_cache()
            torch.npu.synchronize()

        warmup_times = 15
        for _ in range(warmup_times):
            res = op(*args, **kwargs)  # noqa: F841
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif flag_gems.vendor_name == "ascend":
                torch.npu.synchronize()

        test_times = 200
        total_cost = 0.0
        result = None

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            for _ in range(test_times):
                start_event.record()
                result = op(*args, **kwargs)
                end_event.record()
                torch.cuda.synchronize()
                total_cost += start_event.elapsed_time(end_event)
            avg_ms = total_cost / test_times
            avg_sec = avg_ms / 1000.0
        else:
            if flag_gems.vendor_name == "ascend":
                torch.npu.synchronize()
            for _ in range(test_times):
                t0 = time.perf_counter()
                result = op(*args, **kwargs)
                if flag_gems.vendor_name == "ascend":
                    torch.npu.synchronize()
                t1 = time.perf_counter()
                total_cost += t1 - t0
            avg_sec = total_cost / test_times

        return result, avg_sec


@pytest.mark.meshgrid
def test_meshgrid():
    bench = MeshgridBenchmark(
        op_name="meshgrid",
        torch_op=MeshgridBenchmark._torch_meshgrid_with_copy,
        gems_op=flag_gems.meshgrid,
        dtype=consts.FLOAT_DTYPES,
    )
    bench.run()
