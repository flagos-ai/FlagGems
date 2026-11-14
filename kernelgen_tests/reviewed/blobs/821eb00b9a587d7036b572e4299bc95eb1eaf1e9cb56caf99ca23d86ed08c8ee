import pytest
import torch

from . import base, consts, utils


def norm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    p = 2
    yield inp, p


def norm_scalar_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        builder_args=(shape,),
    )


def norm_scalar_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return (utils.generate_tensor_input(shape, dtype, device),)


def norm_scalaropt_dim_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"p": 2, "dim": -1, "keepdim": False},
        builder_args=(shape,),
    )


def norm_scalaropt_dim_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, 2, -1, False


@pytest.mark.norm
def test_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=norm_input_fn,
        op_name="norm",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.norm_scalar
def test_norm_scalar():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=norm_scalar_case_fn,
        build_inputs_fn=norm_scalar_build_inputs_fn,
        op_name="norm_scalar",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.norm_scalaropt_dim
def test_norm_scalaropt_dim():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=norm_scalaropt_dim_case_fn,
        build_inputs_fn=norm_scalaropt_dim_build_inputs_fn,
        op_name="norm_scalaropt_dim",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
