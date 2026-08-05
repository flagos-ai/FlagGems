import pytest
import torch

from . import base

_FLOAT_PARAMS = {}

# Shapes: (seq_len, batch, input_size, hidden_size) - hardcoded because GRU
# requires 4D configuration not covered by core_shapes.yaml
QUANTIZED_GRU_SHAPES = [
    (1, 4, 10, 20),
    (10, 8, 32, 64),
    (32, 16, 64, 128),
    (64, 32, 128, 256),
]


class QuantizedGruBenchmark(base.Benchmark):
    """GRU benchmark comparing FlagGems Triton kernel vs cuDNN GRU baseline.

    Since aten::quantized_gru has no CUDA implementation, we use aten::gru
    (which uses cuDNN) as the baseline for fair comparison.
    """

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        from flag_gems.ops.quantized_gru import quantized_gru_input

        self.gems_op = quantized_gru_input

    def set_shapes(self, shape_file_path=None):
        self.shapes = QUANTIZED_GRU_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


def quantized_gru_input_fn(shape, dtype, device):
    seq_len, batch, input_size, hidden_size = shape

    # CellParamsBase remains CPU-side.  The original floating-point parameters
    # are retained separately for the cuDNN baseline.
    gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
    ).eval()
    gru.qconfig = torch.ao.quantization.default_dynamic_qconfig
    qgru = torch.ao.nn.quantized.dynamic.GRU.from_float(gru)
    params = [module.param for module in qgru._all_weight_values]
    _FLOAT_PARAMS[id(params[0])] = [
        parameter.detach().to(device=device, dtype=dtype)
        for parameter in gru.parameters()
    ]

    # Create input and initial hidden state
    input_tensor = torch.randn(seq_len, batch, input_size, dtype=dtype, device=device)
    hx = torch.randn(1, batch, hidden_size, dtype=dtype, device=device)

    yield (
        {
            "input": input_tensor,
            "hx": hx,
            "params": params,
            "has_biases": True,
            "num_layers": 1,
            "dropout": 0.0,
            "train": False,
            "bidirectional": False,
            "batch_first": False,
        },
    )


def torch_gru_baseline(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    float_params = _FLOAT_PARAMS[id(params[0])]
    return torch.ops.aten.gru.input(
        input,
        hx,
        float_params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    )


@pytest.mark.quantized_gru
def test_perf_quantized_gru():
    bench = QuantizedGruBenchmark(
        input_fn=quantized_gru_input_fn,
        op_name="quantized_gru",
        # Use aten::gru (cuDNN) as baseline since quantized_gru has no CUDA impl
        torch_op=torch_gru_baseline,
        dtypes=[torch.float32, torch.float16],
    )
    bench.run()
