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

import ctypes
import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

from flag_gems.utils.code_cache import code_cache_dir


@lru_cache(maxsize=1)
def _compiled_binary():
    """Build the dav-3510 microkernel once per CANN toolchain and source version."""
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        compiler = shutil.which("bisheng")
        if not compiler:
            return None
        home = str(Path(compiler).parent.parent)
    root = Path(home).resolve()
    arch = root / f"{platform.machine()}-linux"
    cc = arch / "bin" / "bisheng"
    if not cc.is_file():
        return None
    source = Path(__file__).with_name("moe_topk_gate_forward_a5.asc")
    try:
        flags = [
            "-x",
            "asc",
            "--npu-arch=dav-3510",
            "--cce-auto-infer-kernel-type=false",
            "-O2",
            "-std=c++17",
            "-fPIC",
            "--cce-aicore-only",
            "-mllvm",
            "-cce-aicore-dcpreload-args=false",
            *(
                f"-I{p}"
                for p in (
                    arch / "asc/include",
                    arch / "asc/impl",
                    root / "include",
                    root / "pkg_inc",
                )
            ),
        ]
        version = subprocess.check_output(
            [str(cc), "--version"], stderr=subprocess.STDOUT
        )
        key = hashlib.sha256(
            source.read_bytes() + version + repr(flags).encode()
        ).hexdigest()
        cache = code_cache_dir() / "moe_topk_gate_forward_a5" / key
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "moe_topk_gate_forward.aibin"
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.is_file():
                with tempfile.TemporaryDirectory(dir=cache) as tmp:
                    built = Path(tmp) / binary.name
                    obj = Path(tmp) / "route.o"
                    subprocess.run(
                        [str(cc), *flags, str(source), "-o", str(obj)],
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [
                            str(arch / "bin/ld.lld"),
                            "-m",
                            "aicorelinux",
                            "-Ttext",
                            "0",
                            "--no-mmap-output-file",
                            str(obj),
                            "-o",
                            str(built),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    os.replace(built, binary)
        return binary
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(
            f"Atlas A5 moe_topk_gate_forward compiler/library unavailable; using Triton: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


class _LaunchAttr(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("value", ctypes.c_uint32 * 4)]


class _LaunchConfig(ctypes.Structure):
    _fields_ = [("attrs", ctypes.POINTER(_LaunchAttr)), ("numAttrs", ctypes.c_size_t)]


class _KernelArgs(ctypes.Structure):
    _fields_ = [
        (s, ctypes.c_void_p) for s in ("logits", "bias", "indices", "weights")
    ] + [("rows", ctypes.c_int32), ("scale", ctypes.c_float)]


@lru_cache(maxsize=1)
def _runtime():
    rt = ctypes.CDLL("libascendcl.so")
    ptr = ctypes.c_void_p
    rt.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(ptr)]
    rt.aclrtBinaryLoadFromFile.argtypes = [ctypes.c_char_p, ptr, ctypes.POINTER(ptr)]
    rt.aclrtBinaryGetFunction.argtypes = [ptr, ctypes.c_char_p, ctypes.POINTER(ptr)]
    rt.aclrtBinaryUnLoad.argtypes = [ptr]
    rt.aclrtLaunchKernelWithHostArgs.argtypes = [
        ptr,
        ctypes.c_uint32,
        ptr,
        ctypes.POINTER(_LaunchConfig),
        ptr,
        ctypes.c_size_t,
        ptr,
        ctypes.c_size_t,
    ]
    return rt


def _check(status):
    if status:
        raise RuntimeError(f"Ascend C gate kernel ACL status {status}")


@lru_cache(maxsize=None)
def _bound_function(binary, context):
    handle, func = ctypes.c_void_p(), ctypes.c_void_p()
    rt = _runtime()
    _check(rt.aclrtBinaryLoadFromFile(os.fsencode(binary), None, ctypes.byref(handle)))
    status = rt.aclrtBinaryGetFunction(handle, b"moe_score_route", ctypes.byref(func))
    if status:
        rt.aclrtBinaryUnLoad(handle)
        _check(status)
    return handle, func


def launch(tensors, dimensions):
    import torch

    binary = _compiled_binary()
    if binary is None:
        return False
    rows, scale = dimensions
    with torch.npu.device(tensors[0].device):
        rt = _runtime()
        ctx = ctypes.c_void_p()
        _check(rt.aclrtGetCurrentContext(ctypes.byref(ctx)))
        _, func = _bound_function(binary, ctx.value)
        # ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE must cover both slots for SIMT.
        attr = _LaunchAttr()
        attr.id = 2
        attr.value[0] = 160 * 1024
        config = _LaunchConfig(ctypes.pointer(attr), 1)
        args = _KernelArgs(*(t.data_ptr() for t in tensors), rows, scale)
        tile_rows = 8 if rows <= 512 else 32
        stream = torch.npu.current_stream(tensors[0].device).npu_stream
        _check(
            rt.aclrtLaunchKernelWithHostArgs(
                func,
                min(64, (rows + tile_rows - 1) // tile_rows),
                stream,
                ctypes.byref(config),
                ctypes.byref(args),
                ctypes.sizeof(args),
                None,
                0,
            )
        )
    return True
