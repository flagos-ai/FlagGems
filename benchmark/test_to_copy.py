from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config

fp64_is_supported = flag_gems.runtime.device.support_fp64


class ToCopyBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, src_dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_dtype = src_dtype
        if src_dtype is None:
            self._init_type_pairs()

    def _init_type_pairs(self):
        base_dtypes = [torch.float16, torch.bfloat16]
        if fp64_is_supported:
            base_dtypes.append(torch.float64)
        float_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
        uint_dtypes = [torch.uint8]

        self._type_pairs = []

        def add_pairs(src_list, dst_list, skip_same=False):
            for src in src_list:
                for dst in dst_list:
                    if skip_same and src == dst:
                        continue
                    self._type_pairs.append((src, [dst]))

        add_pairs(float_dtypes, base_dtypes, skip_same=True)
        add_pairs(float_dtypes, int_dtypes)
        add_pairs(float_dtypes, uint_dtypes)
        add_pairs(int_dtypes, float_dtypes)
        add_pairs(int_dtypes, int_dtypes, skip_same=True)

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if self.src_dtype in [
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.float64,
            ]:
                inp = torch.randn(shape, dtype=self.src_dtype, device=self.device)
            elif self.src_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                inp = torch.randint(
                    -100, 100, shape, dtype=self.src_dtype, device=self.device
                )
            elif self.src_dtype == torch.uint8:
                inp = torch.randint(
                    0, 255, shape, dtype=self.src_dtype, device=self.device
                )
            else:
                inp = torch.randn(shape, dtype=self.src_dtype, device=self.device)
            yield inp, {"dtype": dtype}

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()

    def run(self):
        if Config.query:
            return super().run()

        if not hasattr(self, "_type_pairs"):
            return super().run()

        self.init_user_config()
        for src_dtype, dst_dtypes in self._type_pairs:
            self.src_dtype = src_dtype
            self.to_bench_dtypes = dst_dtypes
            self._bench_dtype_loop()


@pytest.mark.to_copy
def test_to_copy():
    bench = ToCopyBenchmark(
        op_name="to_copy",
        torch_op=torch.ops.aten._to_copy,
    )
    bench.run()
