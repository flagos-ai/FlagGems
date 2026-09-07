# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build inline TopK fragments and adapt their vector ABI to CANN 9.0."""

import hashlib
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def _normalize(text):
    text = re.sub(
        r"(bufferization\.to_tensor[^\n]*?: memref<[^\n]+>) to tensor<[^\n]+>",
        r"\1",
        text,
    )
    text = re.sub(
        r"operandSegmentSizes = array<i32: (\d+), (\d+), 0>",
        r"operandSegmentSizes = array<i32: \1, \2>",
        text,
    )
    return re.sub(r"\s*tmps\(\s*\)", "", text)


def _split_types(text):
    result, start, depth = [], 0, 0
    for i, char in enumerate(text):
        if char in "(<":
            depth += 1
        elif char in ")>":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    return result + [text[start:].strip()]


def lower(text):
    """CANN prints bufferized CustomOps on one line; only TopK's AIV calls apply."""
    lines, declarations, serial = [], [], 0
    for line in _normalize(text).splitlines(keepends=True):
        if 'symbol = "topk_' not in line or "hivm.hir.custom" not in line:
            lines.append(line)
            continue
        match = re.match(
            r'^(\s*)hivm\.hir\.custom .*symbol = "(topk_\w+)".* ins\((.*?)\) outs\((.*?)\)',
            line,
        )
        if match is None:
            raise ValueError("Expected a bufferized TopK CustomOp from CANN 9.0")
        indent, symbol, ins, outs = match.groups()
        operands, types = [], []
        for fragment in (ins, outs):
            values, value_types = fragment.split(" : ", 1)
            operands.extend(_split_types(values))
            types.extend(_split_types(value_types))
        call_args, call_types = [], []
        for operand, ty in zip(operands, types):
            if ty.startswith("memref"):
                ptr, addr = f"%topk_ptr{serial}", f"%topk_addr{serial}"
                serial += 1
                lines.append(
                    f"{indent}{ptr} = memref.extract_aligned_pointer_as_index {operand} : {ty} -> index\n"
                )
                lines.append(
                    f"{indent}{addr} = arith.index_cast {ptr} : index to i64\n"
                )
                call_args.append(addr)
                call_types.append("i64")
            else:
                call_args.append(operand)
                call_types.append(ty)
        signature = ", ".join(call_types)
        lines.append(
            f"{indent}func.call @_mlir_ciface_{symbol}({', '.join(call_args)}) : ({signature}) -> ()\n"
        )
        declaration = (
            f"  func.func private @_mlir_ciface_{symbol}({signature}) attributes {{"
            "hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, "
            "hivm.tcore_type = #hivm.tcore_type<VECTOR>}\n"
        )
        if declaration not in declarations:
            declarations.append(declaration)
    result = "".join(lines)
    end = result.rfind("}")
    return result[:end] + "".join(declarations) + result[end:]


_HIVMC = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path
sys.path.insert(0, @HELPER_DIR@)
from compile_topk import lower
for arg in sys.argv[1:]:
    if arg.endswith(".mlir") and Path(arg).is_file():
        path = Path(arg)
        text = path.read_text()
        if 'symbol = "topk_' in text and "hivm.hir.custom" in text:
            path.write_text(lower(text))
real = @HIVMC@
os.execv(real, [real, *sys.argv[1:]])
"""
_BISHENG = """#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HERE:$PATH"
exec -a "$0" @COMPILER@ "$@"
"""


def _install(cache, toolkit):
    from triton.backends.ascend import compiler, utils

    if getattr(compiler.linalg_to_bin_enable_npu_compile_A2_A3, "_topk_custom", False):
        return
    # CANN also probes the wrapper without the environment returned by Triton.
    # Bake in the activated toolkit paths for those subprocesses.
    for name, script in (("hivmc", _HIVMC), ("bishengir-compile", _BISHENG)):
        path = cache / name
        path.write_text(
            script.replace("@HELPER_DIR@", repr(str(Path(__file__).parent)))
            .replace("@HIVMC@", repr(str(toolkit / "bin/hivmc")))
            .replace("@COMPILER@", shlex.quote(str(toolkit / "bin/bishengir-compile")))
        )
        path.chmod(0o755)
    original_bc = compiler.linalg_to_bc_by_triton_mlir_opt
    original_linalg = compiler.bc_to_linalg_by_bishengir_opt
    original_bin = compiler.linalg_to_bin_enable_npu_compile_A2_A3
    original_path = compiler._get_npucompiler_path
    tag = b"flaggems-topk-linalg:"

    def to_bc(linalg, metadata, options):
        if 'symbol = "topk_' in linalg and "hivm.hir.custom" in linalg:
            return tag + linalg.encode()
        return original_bc(linalg, metadata, options)

    def to_linalg(data, metadata, options):
        if data.startswith(tag):
            return _normalize(data[len(tag) :].decode())
        return original_linalg(data, metadata, options)

    def get_compiler():
        _, env = original_path()
        env = dict(env)
        env["PATH"] = str(cache) + ":" + env.get("PATH", "")
        return str(cache / "bishengir-compile"), env

    def to_bin(linalg, metadata, options):
        if 'symbol = "topk_' not in linalg or "hivm.hir.custom" not in linalg:
            return original_bin(linalg, metadata, options)
        blockify = compiler._is_auto_map_parallel_blocks_enabled
        compiler._is_auto_map_parallel_blocks_enabled = lambda: False
        try:
            return original_bin(_normalize(linalg), metadata, options)
        finally:
            compiler._is_auto_map_parallel_blocks_enabled = blockify

    to_bin._topk_custom = True
    compiler.linalg_to_bc_by_triton_mlir_opt = to_bc
    compiler.bc_to_linalg_by_bishengir_opt = to_linalg
    compiler.linalg_to_bin_enable_npu_compile_A2_A3 = to_bin
    compiler._get_npucompiler_path = get_compiler
    utils._get_npucompiler_path = get_compiler


def build(source):
    """Use the activated CANN toolkit; generated files stay outside the package."""
    toolkit = Path(os.environ["ASCEND_HOME_PATH"])
    revision = hashlib.sha256(
        source.read_bytes() + Path(__file__).read_bytes()
    ).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / f"flaggems_topk_{os.getuid()}_{revision}"
    cache.mkdir(exist_ok=True)
    include = toolkit / "aarch64-linux/tikcpp/tikcfw"
    cxx = max(Path("/usr/include/c++").iterdir(), key=lambda p: int(p.name))
    command = [
        str(toolkit / "bin/ccec"),
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-aicore-only",
        "-std=c++17",
        "-O3",
        f"-I{include}",
        f"-I{include / 'interface'}",
        f"-I{include / 'impl'}",
        f"-I{cxx}",
        f"-I/usr/include/aarch64-linux-gnu/c++/{cxx.name}",
        "-emit-llvm",
        "-c",
    ]
    bitcode = cache / "topk.bc"
    if not bitcode.exists():
        subprocess.check_call([*command, str(source), "-o", str(bitcode)])
    _install(cache, toolkit)
    return bitcode, shlex.join(command) + " $< -o $@", revision
