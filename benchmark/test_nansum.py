import pytest
import torch

import flag_gems

from . import base, consts

DTYPES = consts.FLOAT_DTYPES + [torch.int8, torch.uint8]


class _NansumBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if cur_dtype.is_floating_point:
                x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 10
                mask = torch.rand(shape, device=self.device) > 0.7
                x[mask] = float("nan")
            else:
                # integers hold no NaN; no int8/uint8 randint on device
                info = torch.iinfo(cur_dtype)
                x = torch.randint(
                    info.min, info.max, shape, dtype=torch.int64, device="cpu"
                ).to(self.device, cur_dtype)

            ndim = x.ndim

            yield (x,)

            if ndim >= 2:
                yield x, -1
                yield x, 0

            if ndim >= 3:
                yield x, 1


@pytest.mark.nansum
def test_benchmark_nansum():
    bench = _NansumBenchmark(
        op_name="nansum",
        torch_op=torch.nansum,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.nansum)
    bench.run()
