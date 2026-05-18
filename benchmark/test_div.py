import pytest
import torch

from . import base, consts, utils


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES (Issue #2693).
@pytest.mark.div_tensor
def test_div():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_tensor_
def test_div_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_tensor_",
        torch_op=lambda a, b: a.div_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _div_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.div_scalar
def test_div_scalar():
    bench = base.GenericBenchmark(
        input_fn=_div_scalar_input_fn,
        op_name="div_scalar",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.div_scalar_
def test_div_scalar_():
    bench = base.GenericBenchmark(
        input_fn=_div_scalar_input_fn,
        op_name="div_scalar_",
        torch_op=lambda a, b: a.div_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _div_scalar_mode_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5, {"rounding_mode": "floor"}


@pytest.mark.div_scalar_mode
def test_div_scalar_mode():
    bench = base.GenericBenchmark(
        input_fn=_div_scalar_mode_input_fn,
        op_name="div_scalar_mode",
        torch_op=torch.div,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _div_scalar_mode_inplace_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.001, {"rounding_mode": None}


@pytest.mark.div_scalar_mode_
def test_div_scalar_mode_():
    bench = base.GenericBenchmark(
        input_fn=_div_scalar_mode_inplace_input_fn,
        op_name="div_scalar_mode_",
        torch_op=lambda a, scalar, rounding_mode=None: a.div_(
            scalar, rounding_mode=rounding_mode
        ),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_div_tensor_mode_(rounding_mode):
    if rounding_mode in ("trunc", "floor"):
        pytest.xfail(
            "Operator bug: trunc/floor div kernels fail Triton compilation for float dtypes"
        )
    bench = base.BinaryPointwiseBenchmark(
        op_name=f"div_tensor_mode_({rounding_mode})",
        torch_op=lambda a, b: a.div_(b, rounding_mode=rounding_mode),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _div_out_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, out


@pytest.mark.div_out
def test_div_out():
    bench = base.GenericBenchmark(
        input_fn=_div_out_input_fn,
        op_name="div_out",
        torch_op=lambda a, b, out: torch.div(a, b, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
