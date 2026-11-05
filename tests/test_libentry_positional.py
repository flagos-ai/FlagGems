# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Argument binding regression coverage for cached LibEntry launches."""

import importlib

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import LibEntry

entry_mod = importlib.import_module("flag_gems.utils.libentry")


@triton.jit(do_not_specialize=["n", "scale"])
def affine_kernel(X, Y, n, scale, BLOCK: tl.constexpr = 128, SHIFT: tl.constexpr = 1):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < n, other=0).to(tl.float32)
    tl.store(Y + offsets, values * scale + SHIFT, offsets < n)


@pytest.mark.parametrize("style", ["positional", "mixed", "keywords", "defaults"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("callable_grid", [False, True])
def test_cached_launch_rebinds_values_and_outputs(style, dtype, callable_grid):
    """Warm calls must use new pointers/runtime values in every binding style."""
    entry = LibEntry(affine_kernel)
    compiled = None
    previous = []
    for index, n in enumerate([65, 257, 130]):
        host = (torch.arange(n, dtype=torch.float32) % 16) / 16 + index
        x = host.to(device=flag_gems.device, dtype=dtype)
        y = torch.full_like(x, -100)
        scale = float(index + 2)
        shift = 1 if style == "defaults" else 7
        grid = (
            (lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),))
            if callable_grid
            else (triton.cdiv(n, 128),)
        )
        with torch_device_fn.device(x.device):
            if style == "positional":
                result = entry[grid](x, y, n, scale, 128, shift, num_warps=4)
            elif style == "mixed":
                result = entry[grid](
                    x, y, n, scale, BLOCK=128, SHIFT=shift, num_warps=4
                )
            elif style == "keywords":
                result = entry[grid](
                    X=x, Y=y, n=n, scale=scale, BLOCK=128, SHIFT=shift, num_warps=4
                )
            else:
                result = entry[grid](x, y, n, scale, num_warps=4)
        if compiled is not None:
            assert result[0] is compiled
        compiled = result[0]
        expected = (host * scale + shift).to(dtype)
        torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
        previous.append((y, expected))
        for old_output, old_expected in previous:
            torch.testing.assert_close(old_output.cpu(), old_expected, rtol=0, atol=0)


@pytest.mark.parametrize("minor", [2, 3, 6, 7])
@pytest.mark.parametrize("callable_grid", [False, True])
def test_cached_positional_launch_keeps_grid_hooks_and_precedence(
    monkeypatch, minor, callable_grid
):
    """Test launcher semantics with controlled cached metadata, without a GPU."""

    @triton.jit(do_not_specialize=["n"])
    def kernel(X, n, SHIFT: tl.constexpr):
        pass

    entry = LibEntry(kernel)
    monkeypatch.setattr(entry_mod, "major_version", 3)
    monkeypatch.setattr(entry_mod, "minor_version", minor)
    monkeypatch.setattr(entry_mod.torch_device_fn, "current_device", lambda: "cpu")
    seen = []

    class Compiled:
        def __getitem__(self, grid):
            def launch(*args):
                seen.append(("launch", grid, args))

            return launch

    def hook(meta):
        seen.append(("hook", meta["X"], meta["n"], meta["SHIFT"]))

    compiled = Compiled()
    constexprs = {"SHIFT": 99}
    x = torch.ones(4)
    key = entry.key([x], [4], [7])
    entry._cpu_cache[key] = (
        compiled,
        constexprs,
        {"SHIFT": 77},
        {"SHIFT": 88},
        ((hook, {"SHIFT": 55}),),
    )
    grid = (lambda meta: (meta["SHIFT"],)) if callable_grid else (2,)
    result = entry.run(x, 4, 7, grid=grid)
    assert result == (compiled, constexprs)
    assert seen[0] == ("hook", x, 4, 55)
    assert seen[1][0:2] == ("launch", (99 if callable_grid else 2, 1, 1))
    launched = seen[1][2]
    assert launched[0] is x
    assert launched[1:] == ((4, 7) if 3 <= minor <= 6 else (4,))
