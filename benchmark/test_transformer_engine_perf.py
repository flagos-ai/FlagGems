import pytest
import torch

import flag_gems
from benchmark.attri_util import DEFAULT_METRICS, FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


class TexGluBenchmark(Benchmark):
    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]
    # Triton grid_y is capped at 65535, BLOCK_SIZE_H=64 -> last dim <= 8388480.
    MAX_LAST_DIM = 2 * 64 * 65535

    def set_more_shapes(self):
        # Last dim must be even for GLU operations to split
        special_shapes_2d = [(1024, 2**i) for i in range(1, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(1, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def init_user_config(self):
        super().init_user_config()
        supported = []
        for shape in self.shapes:
            last_dim = shape[-1]
            if last_dim % 2 != 0:
                continue
            if last_dim > self.MAX_LAST_DIM:
                continue
            supported.append(shape)
        if not supported:
            pytest.skip(
                "No geglu shapes satisfy the constraints of FlagGems implementation."
            )
        self.shapes = supported


class TexGluForwardBenchmark(TexGluBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = generate_tensor_input(shape, cur_dtype, self.device)
            # TE GLU APIs typically accept (input, quantizer).
            yield (x, None)

    def get_tflops(self, op, *args, **kwargs):
        # args[0] is the input tensor x
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


class TexGluBackwardBenchmark(TexGluBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)

            out_shape = list(shape)
            out_shape[-1] = out_shape[-1] // 2

            grad_out = torch.randn(out_shape, dtype=cur_dtype, device=self.device)

            yield grad_out, inp, None

    def get_tflops(self, op, *args, **kwargs):
        # args[1] is the original input tensor 'inp'
        inp_shape = list(args[1].shape)
        # Proxy FLOPs estimate: forward + backward cost roughly approximated
        return torch.tensor(inp_shape).prod().item() * 2


glu_forward_ops = [
    ("geglu", "geglu", FLOAT_DTYPES),
    # ("swiglu", "swiglu", FLOAT_DTYPES),
    # ("reglu", "reglu", FLOAT_DTYPES),
]

glu_backward_ops = [
    ("dgeglu", "dgeglu", FLOAT_DTYPES),
    # ("dswiglu", "dswiglu", FLOAT_DTYPES),
    # ("dreglu", "dreglu", FLOAT_DTYPES),
]


def gems_geglu_wrapper(x, *_):
    return flag_gems.geglu(x)


def gems_dgeglu_wrapper(grad_out, inp, *_args, **_kwargs):
    return flag_gems.dgeglu(grad_out, inp)


@pytest.mark.parametrize(
    "op_name, tex_attr_name, dtypes",
    [
        pytest.param(
            name,
            tex_attr,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, tex_attr, dtype in glu_forward_ops
    ],
)
def test_tex_glu_forward_perf(op_name, tex_attr_name, dtypes):
    if not TE_AVAILABLE:
        pytest.skip("TransformerEngine not installed")

    if not hasattr(tex, tex_attr_name):
        pytest.skip(f"Operator {tex_attr_name} not found in transformer_engine")

    te_op = getattr(tex, tex_attr_name)

    bench = TexGluForwardBenchmark(
        op_name=op_name,
        torch_op=te_op,
        dtypes=dtypes,
        gems_op=gems_geglu_wrapper,
    )
    bench.run()


@pytest.mark.parametrize(
    "op_name, tex_attr_name, dtypes",
    [
        pytest.param(
            name,
            tex_attr,
            dtype,
            marks=getattr(pytest.mark, name, None),
        )
        for name, tex_attr, dtype in glu_backward_ops
    ],
)
def test_tex_glu_backward_perf(op_name, tex_attr_name, dtypes):
    if not TE_AVAILABLE:
        pytest.skip("TransformerEngine not installed")

    if not hasattr(tex, tex_attr_name):
        pytest.skip(f"Operator {tex_attr_name} not found in transformer_engine")

    te_op = getattr(tex, tex_attr_name)

    bench = TexGluBackwardBenchmark(
        op_name=op_name,
        torch_op=te_op,
        dtypes=dtypes,
        is_backward=False,
        gems_op=gems_dgeglu_wrapper,
    )
    bench.run()
