# Copyright 2026, The FlagOS Contributors.
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

"""Optional gfx936 STATE kernel; other configurations retain the Triton path."""

import ctypes
import fcntl
import functools
import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import triton

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_state_kernel():
    try:
        target = triton.runtime.driver.active.get_current_target()
        if target.arch != "gfx936":
            return None
        compiler = Path(
            os.environ.get(
                "FLAG_GEMS_HYGON_CLANG", "/opt/flagos/toolchains/aillvm/bin/clang++"
            )
        )
        if not compiler.is_file():
            return None
        rocm = Path(
            os.environ.get("ROCM_PATH", os.environ.get("HIP_PATH", "/opt/rocm"))
        )
        if rocm.name == "hip":
            rocm = rocm.parent
        source = Path(__file__).with_suffix(".cpp")
        flags = [
            "-x",
            "hip",
            "--offload-arch=gfx936",
            f"--rocm-path={rocm}",
            "-O3",
            "-ffast-math",
            "-mllvm=-support-768-vgprs=true",
            "-mllvm=-check-valu-data-forward-hazards=0",
            "-mllvm=-disable-cluster-lds-memops=true",
            "-mllvm=-enable-hcu-approx-func-fp-math=true",
            "-mllvm=-hcu-update-wait-by-reverse-search=true",
            "-mllvm=-hcu-pre-emit-load-store-opt=false",
            "-std=c++17",
            "-fPIC",
            "-shared",
        ]
        identity = (
            source.read_bytes()
            + repr((str(compiler), compiler.stat().st_mtime_ns, flags)).encode()
        )
        digest = hashlib.sha256(identity).hexdigest()
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache = cache / "flag_gems" / "hygon_state_flash" / digest
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        binary = cache / "kernel.so"
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.exists():
                with tempfile.TemporaryDirectory(dir=cache) as build_dir:
                    output = Path(build_dir) / "kernel.so"
                    command = [
                        str(compiler),
                        *flags,
                        str(source),
                        f"-L{rocm / 'lib'}",
                        "-lamdhip64",
                        "-o",
                        str(output),
                    ]
                    env = os.environ.copy()
                    env["HIP_PATH"] = str(rocm)
                    build = subprocess.run(
                        command, env=env, capture_output=True, text=True, timeout=300
                    )
                    if build.returncode:
                        raise RuntimeError(build.stderr[-4000:])
                    output.replace(binary)
        library = ctypes.CDLL(str(binary))
        library.launch_state_flash.argtypes = [ctypes.c_void_p] * 8
        library.launch_state_flash.restype = ctypes.c_int
        library.launch_short_state_flash.argtypes = [ctypes.c_void_p] * 7
        library.launch_short_state_flash.restype = ctypes.c_int
        return library
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.warning("gfx936 STATE kernel unavailable; using Triton: %s", exc)
        return None


def launch_state_kernel(q, k, v, norms, flags, out, lse):
    library = _load_state_kernel()
    if library is None:
        return False
    stream = torch_device_fn.current_stream(q.device).cuda_stream
    pointers = [
        ctypes.c_void_p(x.data_ptr()) for x in (q, k, v, norms, flags, out, lse)
    ]
    status = library.launch_state_flash(*pointers, ctypes.c_void_p(stream))
    if status:
        raise RuntimeError(f"gfx936 STATE FlashAttention launch failed: HIP {status}")
    return True


def launch_short_state_kernel(q, k, v, flags, out, lse):
    library = _load_state_kernel()
    if library is None:
        return False
    stream = torch_device_fn.current_stream(q.device).cuda_stream
    pointers = [ctypes.c_void_p(x.data_ptr()) for x in (q, k, v, flags, out, lse)]
    status = library.launch_short_state_flash(*pointers, ctypes.c_void_p(stream))
    if status:
        raise RuntimeError(
            f"gfx936 short STATE FlashAttention launch failed: HIP {status}"
        )
    return True
