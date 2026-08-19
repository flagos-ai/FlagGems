import pytest
import torch

import flag_gems

from . import base, consts

# conv_tbc benchmark shapes: (T, B, Cin, Cout, kW, pad). Sizes span typical
# sequence-model conv configs (long time dim, moderate batch/channels) to
# exercise the tiled-matmul kernel across BLOCK autotune candidates.
CONV_TBC_BENCH_SHAPES = [
    (128, 32, 128, 128, 3, 1),
    (256, 16, 256, 256, 3, 1),
    (512, 8, 128, 256, 5, 2),
    (64, 64, 512, 512, 3, 1),
    (1024, 4, 64, 128, 7, 3),
]


class ConvTBCBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype):
        for shape in CONV_TBC_BENCH_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def conv_tbc_input_fn(shape, dtype, device):
    T, B, Cin, Cout, kW, pad = shape
    input = torch.randn((T, B, Cin), device=device, dtype=dtype)
    weight = torch.randn((kW, Cin, Cout), device=device, dtype=dtype)
    bias = torch.randn((Cout,), device=device, dtype=dtype)
    yield {
        "input": input,
        "weight": weight,
        "bias": bias,
        "pad": pad,
    },


def conv_tbc_out_input_fn(shape, dtype, device):
    T, B, Cin, Cout, kW, pad = shape
    input = torch.randn((T, B, Cin), device=device, dtype=dtype)
    weight = torch.randn((kW, Cin, Cout), device=device, dtype=dtype)
    bias = torch.randn((Cout,), device=device, dtype=dtype)
    Tout = T + 2 * pad - kW + 1
    out = torch.empty((Tout, B, Cout), device=device, dtype=dtype)
    yield input, weight, bias, pad, {"out": out}


@pytest.mark.conv_tbc
def test_conv_tbc():
    bench = ConvTBCBenchmark(
        input_fn=conv_tbc_input_fn,
        op_name="conv_tbc",
        torch_op=torch.conv_tbc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_tbc)
    bench.run()


@pytest.mark.conv_tbc_out
def test_conv_tbc_out():
    bench = ConvTBCBenchmark(
        input_fn=conv_tbc_out_input_fn,
        op_name="conv_tbc_out",
        torch_op=torch.ops.aten.conv_tbc.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_tbc_out)
    bench.run()
