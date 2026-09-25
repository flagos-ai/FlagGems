"""Public operator signatures must accept the corresponding ATen keyword names."""

import ast
import inspect
from pathlib import Path

import pytest
import torch

OPS = Path(__file__).resolve().parents[2] / "src/flag_gems/ops"
CASES = (
    [
        (name, name, name, "")
        for name in (
            "_prelu_kernel_backward",
            "arcsinh_",
            "arctanh_",
            "asinh_",
            "atanh_",
            "digamma_",
            "hardswish_",
            "i0_",
            "lift_fresh_copy",
            "log1p_",
            "logit_",
            "mvlgamma_",
            "prelu",
            "relu6",
            "selu",
            "selu_",
            "sgn_",
            "zero",
        )
    ]
    + [("zero", "zero_out", "zero", "out")]
    + [
        (
            f"_upsample_nearest_exact{dim}d",
            f"_upsample_nearest_exact{dim}d{suffix}",
            f"_upsample_nearest_exact{dim}d",
            overload,
        )
        for dim in (1, 3)
        for suffix, overload in (("", ""), ("_out", "out"), ("_vec", "vec"))
    ]
)


@pytest.mark.parametrize("module,name,op,overload", CASES)
def test_aten_signature(module, name, op, overload):
    tree = ast.parse((OPS / f"{module}.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    # Compile only the signature; kernels and module imports are not executed.
    function.decorator_list = []
    function.body = [ast.Pass()]
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<signature>",
            "exec",
        ),
        namespace,
    )
    signature = inspect.signature(namespace[name])
    schema = getattr(torch.ops.aten, op)._schemas[overload]
    assert list(signature.parameters) == [arg.name for arg in schema.arguments]
    kwargs = {arg.name: object() for arg in schema.arguments}
    signature.bind(**kwargs)
    for arg in schema.arguments:
        parameter = signature.parameters[arg.name]
        assert (parameter.kind == inspect.Parameter.KEYWORD_ONLY) == arg.kwarg_only
        if arg.has_default_value():
            assert parameter.default == arg.default_value
