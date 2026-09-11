# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile the FixPipe VDEQF16 fragment for Common IR ``al.custom``.

TLE owns the INT8 cube GEMM. This bitcode only copies scales into the FP
Buffer and drains L0C with FixPipe.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "fixpipe_w8a8_mm.cpp"
_ARTIFACT = _DIR / "fixpipe_w8a8_mm.bc"
_FATBIN = _DIR / "fixpipe_w8a8_mm.bin"
_SYMBOL = "fixpipe_vdeqf16"
_ENTRY = "fixpipe_w8a8_mm_entry_mix_aic"
_ENTRY_AIV = "fixpipe_w8a8_mm_entry_mix_aiv"


def _find_ccec() -> str:
    env = os.environ.get("CCEC") or os.environ.get("BISHENG")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ccec") or shutil.which("bisheng")
    if found:
        return found
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if toolkit:
        cand = Path(toolkit) / "compiler" / "ccec_compiler" / "bin" / "ccec"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("ccec/bisheng not found; source the CANN set_env script")


def _find_ld_lld() -> str:
    env = os.environ.get("CCEC_LINKER")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ld.lld")
    if found:
        return found
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if toolkit:
        cand = Path(toolkit) / "compiler" / "ccec_compiler" / "bin" / "ld.lld"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("ld.lld not found; source the CANN set_env script")


def _tikcpp_include() -> Path:
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if toolkit:
        tik = Path(toolkit) / "aarch64-linux" / "tikcpp" / "tikcfw"
        if (tik / "kernel_operator.h").is_file():
            return tik
    tik = Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/tikcfw")
    if (tik / "kernel_operator.h").is_file():
        return tik
    raise FileNotFoundError("tikcpp/tikcfw/kernel_operator.h not found")


def _cxx_includes() -> list[str]:
    incs: list[str] = []
    for ver in ("12", "11", "13"):
        base = Path(f"/usr/include/c++/{ver}")
        if (base / "cstdint").is_file():
            incs.extend([f"-I{base}", f"-I/usr/include/aarch64-linux-gnu/c++/{ver}"])
            break
    return incs


def compile_cmd(src: Path | None = None, out: Path | None = None) -> list[str]:
    src = src or _SRC
    out = out or _ARTIFACT
    tik = _tikcpp_include()
    return [
        _find_ccec(),
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-aicore-only",
        "-std=c++17",
        f"-I{tik}",
        f"-I{tik / 'interface'}",
        f"-I{tik / 'impl'}",
        *_cxx_includes(),
        "-DFLAGGEMS_COMMON_IR_IFACE",
        # CANN 9.0 rejects ``-emit-llvm`` unless ``-c`` is also present.
        # The object is real LLVM bitcode (magic BC\\xc0\\xde), which
        # ``--link-aicore-bitcode`` can consume. Without ``-emit-llvm``
        # ccec writes a cube ELF relocatable that hivmc cannot link.
        "-emit-llvm",
        "-c",
        str(src),
        "-o",
        str(out),
    ]


def makefile_compile() -> str:
    """Recipe for the CustomOp ``compile`` attribute (``$<`` / ``$@``)."""
    tik = _tikcpp_include()
    incs = " ".join(_cxx_includes())
    return (
        f"{_find_ccec()} -x cce --cce-aicore-arch=dav-c220-cube --cce-aicore-only "
        f"-std=c++17 -I{tik} -I{tik / 'interface'} -I{tik / 'impl'} {incs} "
        f"-DFLAGGEMS_COMMON_IR_IFACE -emit-llvm -c $< -o $@"
    )


def _is_llvm_bitcode(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"BC\xc0\xde"
    except OSError:
        return False


def ensure_bitcode() -> Path:
    """Build the AIC LLVM bitcode for Common IR ``al.custom``.

    ``ccec -x cce -emit-llvm -c`` writes real LLVM bitcode. Rebuild when the
    source is newer or the cached artifact is still a cube ELF relocatable.
    """
    if (
        _ARTIFACT.exists()
        and _ARTIFACT.stat().st_mtime >= _SRC.stat().st_mtime
        and _is_llvm_bitcode(_ARTIFACT)
    ):
        return _ARTIFACT
    cmd = compile_cmd(_SRC, _ARTIFACT)
    logger.info("compiling FixPipe W8A8 AscendC: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    if not _ARTIFACT.exists():
        raise RuntimeError(f"ccec did not write {_ARTIFACT}")
    if not _is_llvm_bitcode(_ARTIFACT):
        raise RuntimeError(f"ccec wrote {_ARTIFACT} but it is not LLVM bitcode")
    return _ARTIFACT


def source_path() -> Path:
    return _SRC


def symbol() -> str:
    return _SYMBOL


def entry_symbol() -> str:
    return _ENTRY


def fatbin_path() -> Path:
    return _FATBIN


def _ccec_core_cmd(
    arch: str, src: Path, out: Path, defines: list[str] | None = None
) -> list[str]:
    tik = _tikcpp_include()
    extras = [f"-D{item}" for item in (defines or [])]
    return [
        _find_ccec(),
        "-x",
        "cce",
        f"--cce-aicore-arch={arch}",
        "--cce-aicore-only",
        "-c",
        "-std=c++17",
        f"-I{tik}",
        f"-I{tik / 'interface'}",
        f"-I{tik / 'impl'}",
        *_cxx_includes(),
        *extras,
        str(src),
        "-o",
        str(out),
    ]


def compile_fatbin_cmd(src: Path | None = None, out: Path | None = None) -> list[str]:
    """Mix compile (cube+vec). TPipe on 910B emits AIV work; cube-only hangs."""
    src = src or _SRC
    out = out or _FATBIN
    tik = _tikcpp_include()
    return [
        _find_ccec(),
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220",
        "-c",
        "-std=c++17",
        f"-I{tik}",
        f"-I{tik / 'interface'}",
        f"-I{tik / 'impl'}",
        *_cxx_includes(),
        str(src),
        "-o",
        str(out),
    ]


def _link_mix_device_elf(cube: Path, vec: Path, out: Path) -> None:
    """Link cube/vec relocatables into one mix EXEC (arch 0x1029).

    AscendC splits AIC (cube) and AIV (vec) from the same source. Both carry
    entry stubs; ``ascendc_pack_kernel`` expects host-stub + merged device.o and
    fails on raw relocatables on CANN 9.0. CANN's ``ld.lld -m aicorelinux``
    matches ``merge_obj.sh`` and is what Triton/AscendC use internally.
    """
    cmd = [
        _find_ld_lld(),
        "-m",
        "aicorelinux",
        "-Ttext=0",
        "--allow-multiple-definition",
        str(cube),
        str(vec),
        "-static",
        "-o",
        str(out),
    ]
    logger.info("linking FixPipe mix ELF: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ld.lld did not write mix device ELF {out}")


def ensure_fatbin() -> Path:
    """Build a mix AIC+AIV ELF that the runtime can launch with TPipe."""
    cube = _DIR / "fixpipe_w8a8_mm.cube.o"
    vec = _DIR / "fixpipe_w8a8_mm.vec.o"
    if (
        _FATBIN.exists()
        and _FATBIN.stat().st_mtime >= _SRC.stat().st_mtime
        and cube.exists()
        and vec.exists()
        and _FATBIN.stat().st_mtime >= cube.stat().st_mtime
        and _FATBIN.stat().st_mtime >= vec.stat().st_mtime
    ):
        return _FATBIN
    logger.info("compiling FixPipe cube/vec objects")
    # Official extract_host_stub remaps the mix wrapper so AIC/AIV keep
    # separate TPipe-inlined bodies. Same rename here.
    subprocess.check_call(
        _ccec_core_cmd(
            "dav-c220-cube",
            _SRC,
            cube,
            [
                f"fixpipe_w8a8_mm_entry={_ENTRY}",
                "fixpipe_w8a8_mm_tpipe=fixpipe_w8a8_mm_tpipe_mix_aic",
                "fixpipe_w8a8_mm_tpipe_buf=fixpipe_w8a8_mm_tpipe_buf_mix_aic",
                "fixpipe_w8a8_mm_args=fixpipe_w8a8_mm_args_mix_aic",
            ],
        )
    )
    subprocess.check_call(
        _ccec_core_cmd(
            "dav-c220-vec",
            _SRC,
            vec,
            [
                f"fixpipe_w8a8_mm_entry={_ENTRY_AIV}",
                "fixpipe_w8a8_mm_tpipe=fixpipe_w8a8_mm_tpipe_mix_aiv",
                "fixpipe_w8a8_mm_tpipe_buf=fixpipe_w8a8_mm_tpipe_buf_mix_aiv",
                "fixpipe_w8a8_mm_args=fixpipe_w8a8_mm_args_mix_aiv",
            ],
        )
    )
    _link_mix_device_elf(cube, vec, _FATBIN)
    logger.info("FixPipe mix ELF ready (%s bytes)", _FATBIN.stat().st_size)
    return _FATBIN


# Triton-Ascend prints HIVM CustomOp with 3 operand segments (ins, outs, tmps).
# InferCoreType can assign CUBE to ``hivm.hir.custom`` but not to ``func.call``.
# CANN 9.0 hivmc does not implement CustomOp (still WIP). Keep the op through
# InferCoreType; the hivmc PATH shim lowers it to an i64 ``func.call`` so
# ``--link-aicore-bitcode`` can attach the AscendC object.
_SEGMENT3_RE = re.compile(
    r"operandSegmentSizes\s*=\s*array<i32:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>"
)
_EMPTY_TMPS_RE = re.compile(r"\s*tmps\(\s*\)")
_SYMBOL_RE = re.compile(r'symbol\s*=\s*"([^"]+)"')
_LAST_CUSTOM_LINALG: str | None = None
_PATCHED = False
_DUMP = Path("/tmp/flaggems_fixpipe_ttadapter.mlir")
_HIVMC_IN = Path("/tmp/flaggems_hivmc_input.mlir")
_HIVMC_OUT = Path("/tmp/flaggems_hivmc_rewritten.mlir")
_WRAP_DIR = _DIR / "_hivmc_wrap"


def _split_ins_outs(fragment: str) -> tuple[str, str]:
    """Split ``vals : types`` from an ``ins(...)`` / ``outs(...)`` body."""
    vals, tys = fragment.split(":", 1)
    return vals.strip(), tys.strip()


_BITCODE_RE = re.compile(r'bitcode\s*=\s*"([^"]+)"')


def _extract_balanced(src: str, open_idx: int) -> tuple[str, int]:
    open_c = src[open_idx]
    close_c = {"(": ")", "{": "}", "[": "]"}[open_c]
    depth = 0
    for j in range(open_idx, len(src)):
        ch = src[j]
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return src[open_idx + 1 : j], j + 1
    raise ValueError(f"unbalanced {open_c} in HIVM custom op")


def lower_custom_op_to_call(mlir: str) -> str:
    """Replace ``hivm.hir.custom`` with an i64 ``func.call`` of ``_mlir_ciface_*``.

    CANN 9.0 hivmc does not implement ``hivm.hir.custom``. Passing memref
    descriptors into ``func.call`` fails later: HIVM wraps the AIC body and
    ``llvm.call`` cannot use values defined outside that region. Extract GM
    addresses next to the call (new SSA, new type) so the call stays legal.
    Legacy dummy ``outs`` stay out of the C ABI. Operations declaring
    ``extra_attr="flaggems_pass_outputs=true"`` receive output buffer addresses
    after their inputs. This preserves existing Fixpipe call signatures.
    """
    decls: list[str] = []
    pieces: list[str] = []
    pos = 0
    sink_n = 0
    while True:
        hit = mlir.find("hivm.hir.custom", pos)
        if hit < 0:
            pieces.append(mlir[pos:])
            break
        line_start = mlir.rfind("\n", 0, hit) + 1
        indent_and_assign = mlir[line_start:hit]
        indent_m = re.match(r"^(\s*)", indent_and_assign)
        indent = indent_m.group(1) if indent_m else ""
        pieces.append(mlir[pos:line_start])

        ins_i = mlir.find("ins(", hit)
        outs_i = mlir.find("outs(", hit)
        if ins_i < 0 or outs_i < 0:
            raise ValueError("hivm.hir.custom is missing ins/outs")
        ins_body, after_ins = _extract_balanced(mlir, ins_i + 3)
        outs_i = mlir.find("outs(", after_ins - 1)
        if outs_i < 0:
            raise ValueError("hivm.hir.custom is missing outs")
        _outs_body, after_outs = _extract_balanced(mlir, outs_i + 4)
        cursor = after_outs
        rest_head = mlir[cursor : cursor + 16]
        if rest_head.lstrip().startswith("tmps("):
            tmps_i = mlir.find("tmps(", cursor)
            _tmps_body, cursor = _extract_balanced(mlir, tmps_i + 4)
        attrs_body = ""
        skip = 0
        while cursor + skip < len(mlir) and mlir[cursor + skip] in " \t\n":
            skip += 1
        if cursor + skip < len(mlir) and mlir[cursor + skip] == "{":
            attrs_body, cursor = _extract_balanced(mlir, cursor + skip)
        loc = ""
        loc_i = mlir.find("loc(", cursor)
        newline_i = mlir.find("\n", cursor)
        if loc_i >= 0 and (newline_i < 0 or loc_i < newline_i):
            _loc_body, after_loc = _extract_balanced(mlir, loc_i + 3)
            loc = " " + mlir[loc_i:after_loc]
            cursor = after_loc
        if cursor < len(mlir) and mlir[cursor] == "\n":
            cursor += 1

        symbol_m = _SYMBOL_RE.search(attrs_body) or _SYMBOL_RE.search(mlir[hit:cursor])
        if symbol_m is None:
            raise ValueError("hivm.hir.custom is missing symbol")
        iface = f"_mlir_ciface_{symbol_m.group(1)}"
        ins_vals, ins_tys = _split_ins_outs(ins_body)
        operands = _split_mlir_list(ins_vals)
        types = _split_mlir_list(ins_tys)
        fragment = mlir[hit:cursor]
        if "flaggems_pass_outputs=true" in fragment:
            out_vals, out_tys = _split_ins_outs(_outs_body)
            operands.extend(_split_mlir_list(out_vals))
            types.extend(_split_mlir_list(out_tys))
        if len(operands) != len(types):
            raise ValueError("hivm.hir.custom operand/type count mismatch")
        call_vals: list[str] = []
        call_tys: list[str] = []
        prefix: list[str] = []
        for name, ty in zip(operands, types):
            # L0C acc is still a tensor at HIVM input. Materialize a memref so
            # we can extract the on-chip address; hivmc 9.0 cannot pass tensors
            # or memref descriptors through llvm.call.
            if ty.startswith("tensor"):
                m_name = f"%fp_m{sink_n}"
                sink_n += 1
                memref_ty = "memref" + ty[len("tensor") :]
                # CANN 9.0 only accepts ``to_memref %t : memref<...>``.
                prefix.append(
                    f"{indent}{m_name} = bufferization.to_memref {name} : {memref_ty}\n"
                )
                name, ty = m_name, memref_ty
            if ty.startswith("memref"):
                p_name = f"%fp_p{sink_n}"
                i_name = f"%fp_i{sink_n}"
                sink_n += 1
                prefix.append(
                    f"{indent}{p_name} = memref.extract_aligned_pointer_as_index "
                    f"{name} : {ty} -> index\n"
                )
                prefix.append(
                    f"{indent}{i_name} = arith.index_cast {p_name} : index to i64\n"
                )
                call_vals.append(i_name)
                call_tys.append("i64")
            else:
                call_vals.append(name)
                call_tys.append(ty)
        pieces.extend(prefix)
        pieces.append(
            f"{indent}func.call @{iface}({', '.join(call_vals)}) "
            f": ({', '.join(call_tys)}) -> (){loc}\n"
        )
        # hivmc on CANN 9.0 rejects hivm.vf_mode / hivm.pipe on func.func.
        # i64 callees need an explicit matching core type for hivmc.
        vector_call = "#hivm.tcore_type<VECTOR>" in fragment
        func_core = "AIV" if vector_call else "AIC"
        tensor_core = "VECTOR" if vector_call else "CUBE"
        attrs = [
            f"hivm.func_core_type = #hivm.func_core_type<{func_core}>",
            "hivm.part_of_mix",
            f"hivm.tcore_type = #hivm.tcore_type<{tensor_core}>",
        ]
        decl = (
            f"  func.func private @{iface}({', '.join(call_tys)}) "
            f"attributes {{{', '.join(attrs)}}}"
        )
        if decl not in decls:
            decls.append(decl)
        pos = cursor

    lowered = "".join(pieces)
    if decls:
        end = lowered.rfind("}")
        if end < 0:
            raise ValueError("cannot insert CustomOp callee: no module end")
        lowered = lowered[:end] + "\n".join(decls) + "\n" + lowered[end:]
    return lowered


def rewrite_custom_op_segments(mlir: str) -> str:
    """Flatten 3-element CustomOp segment sizes to CANN 9.0's 2-element form."""

    def _repl(match: re.Match[str]) -> str:
        ins, outs, tmps = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if tmps != 0:
            raise ValueError(
                f"hivm.hir.custom has non-empty tmps={tmps}; cannot lower to CANN 9.0"
            )
        return f"operandSegmentSizes = array<i32: {ins}, {outs}>"

    return _EMPTY_TMPS_RE.sub("", _SEGMENT3_RE.sub(_repl, mlir))


def _split_mlir_list(src: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in src:
        if ch in "(<":
            depth += 1
            buf.append(ch)
        elif ch in ")>":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf).strip())
    return [x for x in items if x]


def _region_defs(src: str) -> set[str]:
    names = set(re.findall(r"%[\w.]+(?=\s*=)", src))
    names.update(re.findall(r"scf\.for\s+(%[\w.]+)", src))
    return names


def _innermost_for_open(mlir: str, pos: int) -> int | None:
    i = pos
    extra = 0
    while i >= 0:
        ch = mlir[i]
        if ch == "}":
            extra += 1
        elif ch == "{":
            if extra == 0:
                window = mlir[max(0, i - 120) : i]
                if "scf.for" in window:
                    return i
            else:
                extra -= 1
        i -= 1
    return None


def _sink_operand(name: str, ty: str, n: list[int], indent: str) -> tuple[str, str]:
    """Materialize ``name`` inside the current region so llvm.call can use it."""
    sunk = f"%fp_sink_{n[0]}"
    n[0] += 1
    if ty.startswith("memref"):
        return sunk, f"{indent}{sunk} = memref.cast {name} : {ty} to {ty}\n"
    if ty in ("i32", "i64", "i16", "i8"):
        zero = f"%fp_sink_{n[0]}"
        n[0] += 1
        return (
            sunk,
            f"{indent}{zero} = arith.constant 0 : {ty}\n"
            f"{indent}{sunk} = arith.addi {name}, {zero} : {ty}\n",
        )
    raise ValueError(f"cannot sink func.call operand {name}: {ty}")


def sink_call_operands(mlir: str) -> str:
    """Clone outer SSA used by ``func.call`` into the enclosing ``scf.for``.

    hivmc's HIVM-to-LLVM pipeline rejects ``llvm.call`` operands defined
    outside the current region. ``hivm.hir.custom`` was allowed to capture;
    ``func.call`` is not.
    """
    out: list[str] = []
    pos = 0
    sink_n = [0]
    call_re = re.compile(
        r"(\s*)func\.call\s+(@\S+)\(([^)]*)\)\s*:\s*\((.*)\)\s*->\s*\(\)"
    )
    while True:
        m = call_re.search(mlir, pos)
        if m is None:
            out.append(mlir[pos:])
            break
        out.append(mlir[pos : m.start()])
        indent, callee, vals, tys = m.group(1), m.group(2), m.group(3), m.group(4)
        loc = ""
        after = m.end()
        if mlir.startswith(" loc(", after):
            _body, after = _extract_balanced(mlir, after + 4)
            loc = mlir[m.end() : after]
        if f"_mlir_ciface_{_SYMBOL}" not in callee:
            out.append(mlir[m.start() : after])
            pos = after
            continue
        operands = _split_mlir_list(vals)
        types = _split_mlir_list(tys)
        if len(operands) != len(types):
            raise ValueError("func.call operand/type count mismatch")
        # Always rematerialize operands next to the call. HIVM-to-LLVM wraps
        # the AIC body in an inner region; function args then sit outside it
        # and llvm.call is rejected. Nearby memref.cast / addi stay with the call.
        sunk_vals: list[str] = []
        prefix: list[str] = []
        for name, ty in zip(operands, types):
            sunk, text = _sink_operand(name, ty, sink_n, indent)
            prefix.append(text)
            sunk_vals.append(sunk)
        out.extend(prefix)
        out.append(
            f"{indent}func.call {callee}({', '.join(sunk_vals)}) "
            f": ({', '.join(types)}) -> (){loc}"
        )
        if after < len(mlir) and mlir[after] != "\n":
            out.append("\n")
        pos = after
    return "".join(out)


_CALL_RE = re.compile(r"(\s*)func\.call\s+(@\S+)\(([^)]*)\)\s*:\s*\((.*)\)\s*->\s*\(\)")
_FOR_RE = re.compile(
    r"scf\.for\s+(%[\w.]+)\s+=\s+(%[\w.]+)\s+to\s+(%[\w.]+)\s+step\s+(%[\w.]+)\s*:\s*(\S+)"
)


def add_for_iter_args_for_calls(mlir: str) -> str:
    """Pass ``func.call`` captures as ``scf.for`` iter_args.

    Block arguments are defined in the loop region, so hivmc's HIVM-to-LLVM
    lowering can form ``llvm.call`` without capturing outer SSA. Identity
    sinks (``addi 0`` / ``memref.cast``) get folded away and do not help.
    """
    out: list[str] = []
    pos = 0
    ia_n = [0]
    while True:
        m = _FOR_RE.search(mlir, pos)
        if m is None:
            out.append(mlir[pos:])
            break
        brace = mlir.find("{", m.end())
        if brace < 0:
            out.append(mlir[pos:])
            break
        body, after = _extract_balanced(mlir, brace)
        if "func.call" not in body:
            out.append(mlir[pos:after])
            pos = after
            continue
        iv = m.group(1)
        local = _region_defs(body) | {iv}
        mapping: dict[str, tuple[str, str]] = {}
        for cm in _CALL_RE.finditer(body):
            for name, ty in zip(
                _split_mlir_list(cm.group(3)), _split_mlir_list(cm.group(4))
            ):
                if name not in local and name not in mapping:
                    mapping[name] = (f"%fp_ia_{ia_n[0]}", ty)
                    ia_n[0] += 1
        if not mapping:
            out.append(mlir[pos:after])
            pos = after
            continue

        def _repl_call(cm: re.Match[str]) -> str:
            indent, callee, vals, tys = (
                cm.group(1),
                cm.group(2),
                cm.group(3),
                cm.group(4),
            )
            ops = [
                mapping.get(name, (name, ""))[0] if name in mapping else name
                for name in _split_mlir_list(vals)
            ]
            return f"{indent}func.call {callee}({', '.join(ops)}) " f": ({tys}) -> ()"

        new_body = _CALL_RE.sub(_repl_call, body)
        ia_names = [new for new, _ty in mapping.values()]
        ia_tys = [ty for _new, ty in mapping.values()]
        indent_m = re.search(r"^[ \t]+", new_body.lstrip("\n"))
        indent = indent_m.group(0) if indent_m else "      "
        yield_line = f"{indent}scf.yield {', '.join(ia_names)} : {', '.join(ia_tys)}\n"
        if re.search(r"scf\.yield\b", new_body):
            new_body = re.sub(
                r"scf\.yield\b([^\n]*)",
                lambda ym: "scf.yield "
                + ", ".join([ym.group(1).split(":", 1)[0].strip(), *ia_names]).strip(
                    " ,"
                )
                + " : "
                + ", ".join(
                    [
                        *(
                            [ym.group(1).split(":", 1)[1].strip()]
                            if ":" in ym.group(1)
                            else []
                        ),
                        *ia_tys,
                    ]
                ),
                new_body,
                count=1,
            )
        else:
            new_body = new_body.rstrip() + "\n" + yield_line
        inits = ", ".join(f"{new} = {old}" for old, (new, _ty) in mapping.items())
        header = (
            f"scf.for {m.group(1)} = {m.group(2)} to {m.group(3)} step {m.group(4)} "
            f"iter_args({inits}) -> ({', '.join(ia_tys)}) : {m.group(5)} "
        )
        out.append(mlir[pos : m.start()])
        out.append(header)
        out.append("{\n")
        if new_body.startswith("\n"):
            out.append(new_body)
        else:
            out.append(new_body if new_body.endswith("\n") else new_body + "\n")
        out.append("}")
        pos = after
    return "".join(out)


def flatten_for_with_calls(mlir: str) -> str:
    """Inline ``scf.for`` bodies that contain the FixPipe ``func.call``.

    Do not touch TLE's own ``nd2nz`` / ``mma_tile`` loops. hivmc cannot lower
    our ``func.call`` when it captures outer SSA; the induction var is bound
    to 0 because each launched block is already one tile.
    """
    out: list[str] = []
    pos = 0
    iv_n = [0]
    while True:
        m = _FOR_RE.search(mlir, pos)
        if m is None:
            out.append(mlir[pos:])
            break
        brace = mlir.find("{", m.end())
        if brace < 0:
            out.append(mlir[pos:])
            break
        body, after = _extract_balanced(mlir, brace)
        if f"_mlir_ciface_{_SYMBOL}" not in body:
            out.append(mlir[pos:after])
            pos = after
            continue
        iv, iv_ty = m.group(1), m.group(5)
        new_iv = f"%fp_iv_{iv_n[0]}"
        iv_n[0] += 1
        body = re.sub(rf"{re.escape(iv)}(?![\w.])", new_iv, body)
        line_start = mlir.rfind("\n", pos, m.start()) + 1
        indent = re.match(r"[ \t]*", mlir[line_start : m.start()]).group(0)
        out.append(mlir[pos : m.start()])
        out.append(f"{indent}{new_iv} = arith.constant 0 : {iv_ty}\n")
        if not body.startswith("\n"):
            out.append("\n")
        out.append(body if body.endswith("\n") else body + "\n")
        pos = after
    return "".join(out)


def _skip_mlir_type(src: str, i: int) -> int:
    while i < len(src) and src[i] in " \t":
        i += 1
    while i < len(src) and (src[i].isalnum() or src[i] in "._!"):
        i += 1
    while i < len(src) and src[i] == "<":
        depth = 0
        while i < len(src):
            if src[i] == "<":
                depth += 1
            elif src[i] == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    return i


def rewrite_cann90_bufferization(mlir: str) -> str:
    """Drop FlagTree's ``to_tensor %x : memref<T> to tensor<T>`` result type.

    CANN 9.0 hivmc only parses ``bufferization.to_tensor %x : memref<T>``.
    """
    key = "bufferization.to_tensor"
    out: list[str] = []
    pos = 0
    while True:
        hit = mlir.find(key, pos)
        if hit < 0:
            out.append(mlir[pos:])
            break
        colon = mlir.find(":", hit)
        newline = mlir.find("\n", hit)
        if colon < 0 or (newline >= 0 and colon > newline):
            out.append(mlir[pos : hit + len(key)])
            pos = hit + len(key)
            continue
        ty_end = _skip_mlir_type(mlir, colon + 1)
        rest = mlir[ty_end : ty_end + 16]
        if rest.lstrip().startswith("to "):
            to_i = mlir.find("to ", ty_end)
            ty_end = _skip_mlir_type(mlir, to_i + 2)
            out.append(mlir[pos:colon])
            out.append(mlir[colon : mlir.find("to ", colon)])
            pos = ty_end
        else:
            out.append(mlir[pos:ty_end])
            pos = ty_end
    return "".join(out)


_CAST_DEF_RE = re.compile(r"^\s*(%[\w.]+)\s*=\s*memref\.cast\s+(%[\w.]+)\s*:")
_PTR_DEF_RE = re.compile(
    r"^\s*(%[\w.]+)\s*=\s*hivm\.hir\.pointer_cast[^:]*:\s*(memref.*)"
)
_NZ2ND_RE = re.compile(r"func\.call\s+@fixpipe_nz2nd_\S+\(")


def _memref_result_type(tail: str) -> str:
    tail = tail.strip()
    if " loc(" in tail:
        tail = tail[: tail.find(" loc(")].strip()
    return tail.rstrip()


def rewire_custom_acc_from_default_fixpipe(mlir: str) -> str:
    """Point CustomOp at L0C and drop HIVM's default L0C→L1 ``fixpipe_nz2nd``.

    TLE cannot see VDEQF16, so HIVM drains ``cc`` into ``cbuf`` first and
    hands the custom the L1 tile. Replace that drain so FixPipe reads L0C.
    """
    if "hivm.hir.custom" not in mlir or "fixpipe_nz2nd_" not in mlir:
        return mlir
    casts: dict[str, str] = {}
    src_ty: dict[str, str] = {}
    for ln in mlir.splitlines():
        m = _CAST_DEF_RE.match(ln)
        if m:
            dst, src = m.group(1), m.group(2)
            casts[dst] = src
            colon = ln.find(":", m.end() - 1)
            if colon >= 0:
                to_i = ln.find(" to ", colon)
                if to_i >= 0:
                    src_ty[src] = _memref_result_type(ln[colon + 1 : to_i])
            continue
        m = _PTR_DEF_RE.match(ln)
        if m:
            src_ty[m.group(1)] = _memref_result_type(m.group(2))

    def _unwrap(name: str) -> str:
        seen: set[str] = set()
        while name in casts and name not in seen:
            seen.add(name)
            name = casts[name]
        return name

    nz_src = nz_dst = None
    for ln in mlir.splitlines():
        if _NZ2ND_RE.search(ln) is None:
            continue
        paren = ln.find("(")
        close = ln.rfind(")")
        ops = _split_mlir_list(ln[paren + 1 : close])
        if len(ops) < 2:
            continue
        nz_src, nz_dst = _unwrap(ops[0]), _unwrap(ops[1])
        break
    if nz_src is None:
        return mlir

    # Drop the default L0C→L1 drain and hold TLE's FIX→M release until
    # after VDEQF16, otherwise the next tile's MMA races FixPipe.
    delayed_fix_m = None
    saw_nz = False
    out: list[str] = []
    for ln in mlir.splitlines(keepends=True):
        if "pipe_barrier[<PIPE_FIX>]" in ln:
            continue
        if _NZ2ND_RE.search(ln):
            saw_nz = True
            continue
        if saw_nz and delayed_fix_m is None and "set_flag[<PIPE_FIX>, <PIPE_M>" in ln:
            delayed_fix_m = ln
            continue
        if "hivm.hir.custom" not in ln or "ins(" not in ln:
            out.append(ln)
            continue
        body = ln.split("ins(", 1)[1]
        vals, rest = body.split(":", 1)
        names = _split_mlir_list(vals)
        tys_part, after_tys = rest.split(")", 1)
        types = _split_mlir_list(tys_part)
        if names and names[0] == nz_dst:
            names[0] = nz_src
            if nz_src in src_ty:
                types[0] = src_ty[nz_src]
            ln = (
                ln.split("ins(", 1)[0]
                + "ins("
                + ", ".join(names)
                + " : "
                + ", ".join(types)
                + ")"
                + after_tys
            )
        out.append(ln)
        if delayed_fix_m is not None:
            out.append(delayed_fix_m)
            delayed_fix_m = None
    if delayed_fix_m is not None:
        out.append(delayed_fix_m)
    return "".join(out)


def strip_declared_cube_only_stub(mlir: str) -> str:
    # Only explicit opt-in kernels whose actual work is entirely in AIC.
    while True:
        m = re.search(r"  func\.func @(\w+)_mix_aiv\(", mlir)
        if m is None:
            break
        line_end = mlir.index("\n", m.start())
        start = mlir.rfind("{", m.start(), line_end)
        body, end = _extract_balanced(mlir, start)
        if "hivm.hir.custom" in body or re.search(
            r"(?:func\.)?call @(?!broadcast_scalar_)", body
        ):
            raise ValueError("cube-only marker would discard a nonempty AIV body")
        if any(
            op in body
            for op in (
                "hivm.hir.load",
                "hivm.hir.store",
                "memref.store",
                "memref.copy",
                "llvm.store",
            )
        ):
            raise ValueError("cube-only marker would discard AIV memory operations")
        mlir = mlir[: m.start()] + mlir[end:]
    mlir = re.sub(r"@(\w+)_mix_aic(?=\()", r"@\1", mlir)
    mlir = re.sub(r",\s*hivm\.part_of_mix\b", "", mlir)
    mlir = re.sub(r"\bhivm\.part_of_mix\s*,\s*", "", mlir)
    mlir = re.sub(r"\bhivm\.part_of_mix\b", "", mlir)
    mlir = re.sub(
        r"hivm\.module_core_type = #hivm\.module_core_type<[^>]+>",
        "hivm.module_core_type = #hivm.module_core_type<AIC>",
        mlir,
    )
    mlir = re.sub(r"^.*hivm\.hir\.sync_block_(?:set|wait).*\n", "", mlir, flags=re.M)
    return mlir


def prepare_hivmc_mlir(mlir: str) -> str:
    """Lower ``hivm.hir.custom`` for CANN 9.0 hivmc (op is unknown there)."""
    cube_only = "flaggems_cube_only=true" in mlir
    rewritten = rewrite_cann90_bufferization(mlir)
    rewritten = rewrite_custom_op_segments(rewritten)
    if 'mix_mode = "aic"' in rewritten:
        # CANN 9.0 can retain an empty AIV stub after splitting a CUBE-only
        # scope and then emit task type 32 (MIX_AIC, ratio 2).  This kernel's
        # useful body and linked bitcode are both AIC-only, so advertise task
        # type 20 (AI_CORE, ratio 0) to prevent the runtime from scheduling
        # two AIV blocks for every Cube block.
        rewritten = rewritten.replace(
            "arith.constant 32 : i8", "arith.constant 20 : i8", 1
        )
    if "hivm.hir.custom" in rewritten:
        rewritten = rewire_custom_acc_from_default_fixpipe(rewritten)
        rewritten = lower_custom_op_to_call(rewritten)
        # Do not flatten: a cube may scan many tiles. Binding the
        # induction var to 0 would rewrite every tile as tile 0.
        rewritten = sink_call_operands(rewritten)
    if cube_only:
        rewritten = strip_declared_cube_only_stub(rewritten)
    return rewritten


def _prepare_custom_linalg(linalg: str) -> str:
    """Keep ``hivm.hir.custom`` so InferCoreType can assign CUBE."""
    if "flaggems_cube_only=true" in linalg:
        if "#hivm.tcore_type<VECTOR>" in linalg:
            raise ValueError(
                "cube-only marker conflicts with a Vector custom operation"
            )
        linalg = re.sub(r'mix_mode\s*=\s*"[^"]+"', 'mix_mode = "aic"', linalg)
    rewritten = rewrite_cann90_bufferization(linalg)
    _DUMP.write_text(rewritten)
    rewritten = rewrite_custom_op_segments(rewritten)
    Path("/tmp/flaggems_fixpipe_ttadapter.rewritten.mlir").write_text(rewritten)
    return rewritten


def _find_real_hivmc() -> str:
    wrap = (_WRAP_DIR / "hivmc").resolve()
    cached = os.environ.get("FLAGGEMS_REAL_HIVMC")
    if cached:
        cand = Path(cached)
        if cand.is_file() and cand.resolve() != wrap:
            return str(cand)
    toolkit = (
        os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or ""
    )
    candidates = [
        "/usr/local/Ascend/cann-9.0.0/bin/hivmc",
        "/usr/local/Ascend/cann-9.0.0/tools/bishengir/bin/hivmc",
    ]
    if toolkit:
        candidates.extend(
            [
                str(Path(toolkit) / "bin" / "hivmc"),
                str(Path(toolkit) / "tools" / "bishengir" / "bin" / "hivmc"),
            ]
        )
    for c in candidates:
        p = Path(c)
        if p.is_file() and p.resolve() != wrap:
            return str(p)
    for d in os.environ.get("PATH", "").split(":"):
        p = Path(d) / "hivmc"
        if p.is_file() and p.resolve() != wrap:
            return str(p)
    raise FileNotFoundError("real hivmc not found")


_HIVMC_SHIM = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from compile_fixpipe import prepare_hivmc_mlir  # noqa: E402

_HIVMC_IN = Path("/tmp/flaggems_hivmc_input.mlir")
_HIVMC_OUT = Path("/tmp/flaggems_hivmc_rewritten.mlir")


def _real_hivmc() -> str:
    real = os.environ.get("FLAGGEMS_REAL_HIVMC")
    if real and Path(real).is_file() and Path(real).resolve() != Path(__file__).resolve():
        return real
    from compile_fixpipe import _find_real_hivmc
    return _find_real_hivmc()


def main() -> None:
    args = sys.argv[1:]
    new_args = []
    for arg in args:
        path = Path(arg)
        if path.suffix == ".mlir" and path.is_file():
            text = path.read_text()
            _HIVMC_IN.write_text(text)
            if "hivm.hir.custom" in text:
                Path("/tmp/flaggems_hivmc_custom_in.mlir").write_text(text)
                text = prepare_hivmc_mlir(text)
                path.write_text(text)
                Path("/tmp/flaggems_hivmc_custom_out.mlir").write_text(text)
            _HIVMC_OUT.write_text(text)
        new_args.append(arg)
    real = _real_hivmc()
    os.execv(real, [real, *new_args])


if __name__ == "__main__":
    main()
"""


_BISHENGIR_SHIM = """#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HERE:${PATH}"
REAL="${FLAGGEMS_REAL_BISHENGIR:-/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile}"
# Keep argv[0] as this wrapper so sibling hivmc lookup hits our shim.
exec -a "$0" "$REAL" "$@"
"""


def _ensure_hivmc_wrapper() -> Path:
    _WRAP_DIR.mkdir(exist_ok=True)
    shim = _WRAP_DIR / "hivmc"
    if not shim.exists() or shim.read_text() != _HIVMC_SHIM:
        shim.write_text(_HIVMC_SHIM)
        shim.chmod(0o755)
    compiler = _WRAP_DIR / "bishengir-compile"
    if not compiler.exists() or compiler.read_text() != _BISHENGIR_SHIM:
        compiler.write_text(_BISHENGIR_SHIM)
        compiler.chmod(0o755)
    return _WRAP_DIR


def install_cann90_custom_op_compat() -> None:
    """Keep CustomOp for InferCoreType; rewrite it only when hivmc starts."""
    global _PATCHED
    if _PATCHED:
        return
    from triton.backends.ascend import compiler as ascend_compiler
    from triton.backends.ascend import utils as ascend_utils

    compile_entry = (
        "_compile_linalg_to_npu_bin"
        if hasattr(ascend_compiler, "_compile_linalg_to_npu_bin")
        else "linalg_to_bin_enable_npu_compile_A2_A3"
    )
    original_compile = getattr(ascend_compiler, compile_entry)
    if getattr(original_compile, "_flaggems_fixpipe", False):
        _PATCHED = True
        return

    wrap_dir = str(_ensure_hivmc_wrapper())
    real_hivmc = _find_real_hivmc()
    orig_to_bc = ascend_compiler.linalg_to_bc_by_triton_mlir_opt
    orig_to_lin = ascend_compiler.bc_to_linalg_by_bishengir_opt
    orig_to_bin = original_compile
    orig_get_compiler = ascend_compiler._get_npucompiler_path

    def _to_bc(linalg, metadata, opt):
        global _LAST_CUSTOM_LINALG
        if "hivm.hir.custom" in linalg:
            _LAST_CUSTOM_LINALG = linalg
            _DUMP.write_text(linalg)
            return b""
        _LAST_CUSTOM_LINALG = None
        return orig_to_bc(linalg, metadata, opt)

    def _to_lin(bc_data, metadata, opt):
        if not bc_data and _LAST_CUSTOM_LINALG:
            return _prepare_custom_linalg(_LAST_CUSTOM_LINALG)
        return orig_to_lin(bc_data, metadata, opt)

    def _wrapped_get_compiler():
        path, env = orig_get_compiler()
        env = dict(env)
        env["PATH"] = wrap_dir + ":" + env.get("PATH", "")
        env["FLAGGEMS_REAL_HIVMC"] = real_hivmc
        env["FLAGGEMS_REAL_BISHENGIR"] = path
        return str(Path(wrap_dir) / "bishengir-compile"), env

    def _to_bin(linalg, metadata, opt):
        if "hivm.hir.custom" in linalg or _SEGMENT3_RE.search(linalg):
            # Every operation in this kernel is inside a CUBE scope and the
            # linked CommonIR fragment is compiled for dav-c220-cube.  The
            # generic TLE lowering still labels the wrapper as ``mix``, which
            # makes the runtime launch two AIV blocks per AIC block even
            # though the AIV body has no useful work.  Mark the wrapper AIC so
            # the launcher submits only the 20 Cube blocks.
            linalg = re.sub(
                r'mix_mode\s*=\s*"mix"', 'mix_mode = "aic"', linalg, count=1
            )
            linalg = _prepare_custom_linalg(linalg)
            # Keep func.call at function scope so hivmc can lower it.
            orig_blockify = ascend_compiler._is_auto_map_parallel_blocks_enabled
            ascend_compiler._is_auto_map_parallel_blocks_enabled = lambda: False
            try:
                return orig_to_bin(linalg, metadata, opt)
            finally:
                ascend_compiler._is_auto_map_parallel_blocks_enabled = orig_blockify
        return orig_to_bin(linalg, metadata, opt)

    _to_bin._flaggems_fixpipe = True
    ascend_compiler.linalg_to_bc_by_triton_mlir_opt = _to_bc
    ascend_compiler.bc_to_linalg_by_bishengir_opt = _to_lin
    setattr(ascend_compiler, compile_entry, _to_bin)
    ascend_compiler._get_npucompiler_path = _wrapped_get_compiler
    ascend_utils._get_npucompiler_path = _wrapped_get_compiler
    _PATCHED = True
