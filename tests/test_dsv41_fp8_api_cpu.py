# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of the block-FP8 public call contract."""

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch
import triton


def public_matmul():
    path = Path(__file__).parents[1] / "src/flag_gems/ops/w8a8_block_fp8_matmul.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "w8a8_block_fp8_matmul"
    )
    for arg in (*function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    function.returns = None
    namespace = {"math": math, "torch": torch, "triton": triton}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


class Tensor:
    def __init__(self, shape, dtype=torch.float32, *, floating=True):
        self.shape, self.dtype = shape, dtype
        self.ndim = len(shape)
        self.is_cuda = False
        self._floating = floating

    def is_floating_point(self):
        return self._floating

    def is_contiguous(self):
        return True

    def new_empty(self, shape, dtype):
        return (tuple(shape), dtype)


def test_public_api_preserves_native_arguments_and_empty_output():
    fn = public_matmul()
    assert list(inspect.signature(fn).parameters) == [
        "A",
        "B",
        "As",
        "Bs",
        "block_size",
        "output_dtype",
    ]
    args = [Tensor((0, 64)), Tensor((32, 64)), Tensor((0, 2)), Tensor((1, 2))]
    assert fn(*args, [32, 32], torch.float32) == ((0, 32), torch.float32)
    with pytest.raises(TypeError, match="tuned"):
        fn(*args, [32, 32], tuned=False)


def test_bad_scale_and_block_size_fail_before_execution():
    fn = public_matmul()
    a, b, sa, sb = (
        Tensor((0, 64)),
        Tensor((32, 64)),
        Tensor((0, 2), floating=False),
        Tensor((1, 2)),
    )
    with pytest.raises(TypeError, match="numeric scales"):
        fn(a, b, sa, sb, [32, 32])
    with pytest.raises(ValueError, match="powers of two"):
        fn(a, b, sa, sb, [32, 48])


def test_tle_capability_is_checked_before_dispatch(monkeypatch):
    import builtins
    import sys
    import types

    path = Path(__file__).parents[1] / "src/flag_gems/ops/w8a8_block_fp8_matmul.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_tle_short"
    )
    package = types.ModuleType("isolated_fp8_ops")
    package.__path__ = []
    short = types.ModuleType("isolated_fp8_ops._w8a8_block_fp8_tle_short")
    fn = object()
    short.block_fp8_matmul_tle_short = fn
    short.tle = types.SimpleNamespace(
        gpu=types.SimpleNamespace(alloc=1, copy=1, smem=1, wgmma=1)
    )
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, short.__name__, short)
    namespace = {"__package__": package.__name__}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    resolve = namespace[function.name]
    assert resolve() is None
    short.tle.gpu.wgmma_wait = 1
    assert resolve() is fn

    original_import = builtins.__import__

    def fail_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_w8a8_block_fp8_tle_short":
            raise ModuleNotFoundError(name="triton.experimental.tle")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    assert resolve() is None

    def fail_unrelated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_w8a8_block_fp8_tle_short":
            raise ModuleNotFoundError(
                "unrelated_dependency", name="unrelated_dependency"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_unrelated_import)
    with pytest.raises(ModuleNotFoundError, match="unrelated_dependency"):
        resolve()
