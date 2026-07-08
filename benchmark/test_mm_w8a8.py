import os
import pytest
import torch
import yaml

import flag_gems

from . import base


def _mm_w8a8_available() -> bool:
    return (
        flag_gems.device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
        and hasattr(torch, "float8_e4m3fn")
    )


def mm_w8a8_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


class MmW8A8Benchmark(base.BlasBenchmark):
    def set_shapes(self, shape_file_path=None):
        shape_file_path = shape_file_path or self.DEFAULT_SHAPE_FILES
        self.shape_desc = "B, M, N, K"
        if not os.path.isfile(shape_file_path):
            raise FileNotFoundError(f"Shape file '{shape_file_path}' does not exist.")

        with open(shape_file_path, "r", encoding="utf-8") as shape_file:
            yaml_config = yaml.safe_load(shape_file) or {}

        shape_key = "mm_w8a8" if "mm_w8a8" in yaml_config else "mm"
        if shape_key in yaml_config:
            self.shapes = [tuple(shape) for shape in yaml_config[shape_key]["shapes"]]
            self.shape_desc = yaml_config[shape_key].get("shape_desc", self.shape_desc)
        else:
            self.shapes = self.DEFAULT_SHAPES

    def get_tflops(self, op, *args, **kwargs):
        return args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2


@pytest.mark.mm_w8a8
@pytest.mark.skipif(
    not _mm_w8a8_available(),
    reason="mm_w8a8 benchmark requires CUDA Hopper FP8 support",
)
def test_mm_w8a8():
    bench = MmW8A8Benchmark(
        op_name="mm_w8a8",
        input_fn=mm_w8a8_input_fn,
        torch_op=torch.mm,
        gems_op=flag_gems.mm_w8a8,
        dtypes=[torch.bfloat16],
    )
    bench.run()
