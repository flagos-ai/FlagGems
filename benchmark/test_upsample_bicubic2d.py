import pytest
import torch

from . import base, consts


# TODO(Qiming): Kill this class
class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


@pytest.mark.skip(reason="Benchmark fails: issue #2666")
@pytest.mark.upsample_bicubic2d
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_bicubic2d(align_corners):
    def _input_fn(shape, dtype, device):
        input = torch.randn(shape, device=device, dtype=dtype)
        scale_factors = [2.0, 2.0]
        output_size = None

        yield {
            "input": input,
            "output_size": output_size,
            "align_corners": align_corners,
            "scale_factors": scale_factors,
        }

    bench = UpsampleBenchmark(
        input_fn=_input_fn,
        op_name="upsample_bicubic2d",
        torch_op=torch._C._nn.upsample_bicubic2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
