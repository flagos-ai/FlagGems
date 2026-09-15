# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""INT8 Cube on prepacked NZ inputs with exact INT32 output and FP32 scales."""

import fcntl
import hashlib
import importlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems.utils import triton_lang_extension as ext

from . import compile_fixpipe as helper

ROOT = Path(__file__).parent
CACHE = Path(tempfile.gettempdir()) / "flaggems_mm_nz"
READY = set()
_REGISTER_LOCK = threading.RLock()


def _register_impl(m, n, k, mp, np, bm, bn, bk, cores):
    assert (bm + bn) * k <= 512 * 1024
    assert bm * bn * 4 <= 128 * 1024
    assert bm * bk <= 65536 and bn * bk <= 65536
    assert k % (2 * bk) == 0
    source = (ROOT / "mm_nz.cpp").read_bytes()
    compiler = Path(helper._find_ccec())
    identity = repr(
        (m, n, k, mp, np, bm, bn, bk, cores, str(compiler), compiler.stat().st_mtime_ns)
    ).encode()
    revision = hashlib.sha256(
        source
        + identity
        + Path(__file__).read_bytes()
        + Path(helper.__file__).read_bytes()
    ).hexdigest()
    name = "mm_nz_" + revision[:24]
    if name in READY:
        return name, revision
    CACHE.mkdir(parents=True, exist_ok=True)
    cpp = CACHE / (name + ".cpp")
    bc = CACHE / (name + ".bc")
    defines = dict(
        M=m,
        N=n,
        K=k,
        MP=mp,
        NP=np,
        BM=bm,
        BN=bn,
        BK=bk,
        CORES=cores,
        CUBE_ENTRY="_mlir_ciface_" + name,
    )
    source_bytes = (
        "".join(
            f"#define CV_{key} {value}\n" for key, value in defines.items()
        ).encode()
        + source
    )
    with (CACHE / (name + ".lock")).open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cpp.exists() or cpp.read_bytes() != source_bytes:
            cpp.write_bytes(source_bytes)
        if not bc.exists() or bc.stat().st_size == 0:
            temporary = bc.with_name(bc.name + f".{os.getpid()}.tmp")
            try:
                subprocess.check_call(
                    helper.compile_cmd(cpp, temporary) + ["-O3", "-I", str(ROOT)]
                )
                os.replace(temporary, bc)
            finally:
                temporary.unlink(missing_ok=True)

    def init(self, a, b, o, pid, out=None):
        assert out is not None
        self.arg_type["pid"] = tl.int32

    op = type(
        name,
        (),
        dict(
            name=name,
            core=al.CORE.CUBE,
            pipe=al.PIPE.PIPE_ALL,
            mode=al.MODE.SIMD,
            symbol=name,
            bitcode=str(bc),
            source=str(cpp),
            extra_attr="flaggems_cube_only=true",
            compile=helper.makefile_compile().replace(
                "-std=c++17", f"-std=c++17 -O3 -I{ROOT}"
            ),
            __init__=init,
        ),
    )
    al.register_custom_op(op)
    READY.add(name)
    return name, revision


def _register(*args):
    with _REGISTER_LOCK:
        return _register_impl(*args)


@triton.jit
def _mm_nz_kernel(A, B, C, OP: tl.constexpr, REV: tl.constexpr):
    with al.scope(core_mode="cube"):
        dummy = tl.load(A + tl.arange(0, 16), mask=False)
        al.custom(OP, A, B, C, ext.program_id(0).to(tl.int32), out=dummy)


def prepare(aq, bq, a_s, b_s, out, m, n, k, tiles):
    bm, bn, bk = tiles
    assert aq.dtype == bq.dtype == torch.int8
    assert a_s.dtype == b_s.dtype == torch.float32
    assert out.dtype in (torch.bfloat16, torch.float16) and out.is_contiguous()
    assert m > 0 and m % 8 == 0 and n % 32 == 0 and k % 32 == 0
    assert bm % 16 == 0 and bn % 16 == 0
    mod = importlib.import_module("flag_gems.runtime.backend._ascend.ops.mm_w8a8_fp8")
    mp, np = triton.cdiv(m, bm) * bm, triton.cdiv(n, bn) * bn
    cores = min(mod._cube_core_count(), (mp // bm) * (np // bn))
    a = torch.zeros((mp, k), device=aq.device, dtype=torch.int8)
    a[:m] = aq
    b = torch.zeros((np, k), device=aq.device, dtype=torch.int8)
    b[:n] = bq.T
    # Each tile is [K/32, M(or N)/16, 16, 32], matching C1 NZ directly.
    a_q = (
        a.reshape(mp // bm, bm // 16, 16, k // 32, 32)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    b_q = (
        b.reshape(np // bn, bn // 16, 16, k // 32, 32)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    a_s, b_s = a_s.contiguous(), b_s.contiguous()
    acc = torch.empty((mp, np), device=aq.device, dtype=torch.int32)
    rows, cols = (16, 1024) if m % 16 == 0 else (8, 1024)
    if n == 12288:
        rows, cols = 8, 2048
    if n == 1024 and m <= 128:
        rows = 8
    if n < 1024:
        cols = min(256, n)
        rows = 32 if m % 32 == 0 else (16 if m % 16 == 0 else 8)
    grid_v = min(40, (m // rows) * (n // cols))
    name, revision = _register(m, n, k, mp, np, bm, bn, bk, cores)

    def call():
        _mm_nz_kernel[(cores,)](a_q, b_q, acc, name, revision, num_warps=1)
        mod._scale_int32_dense_kernel[(grid_v,)](
            acc, a_s, b_s, out, m, n, np, rows, cols
        )

    return call, {
        "path": "nz_int32_vector",
        "tiles": list(tiles),
        "scale_tile": [rows, cols],
        "kernel_count": 2,
        "workspace_bytes": acc.numel() * acc.element_size(),
        "input_layout": "tile_k32_mn16_nz",
        "revision": revision,
    }
