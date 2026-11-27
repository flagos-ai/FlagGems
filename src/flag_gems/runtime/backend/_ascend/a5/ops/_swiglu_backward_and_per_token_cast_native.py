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
def _native_launcher():
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
    source = Path(__file__).with_name("swiglu_backward_and_per_token_cast_a5.asc")
    try:
        flags = [
            "-x",
            "asc",
            "--npu-arch=dav-3510",
            "--cce-auto-infer-kernel-type=false",
            "-O2",
            "-std=c++17",
            "-fPIC",
            "-shared",
            *(
                f"-I{p}"
                for p in (
                    arch / "asc/include",
                    arch / "asc/impl",
                    root / "include",
                    root / "pkg_inc",
                )
            ),
            f"-L{root / 'lib64'}",
        ]
        version = subprocess.check_output(
            [str(cc), "--version"], stderr=subprocess.STDOUT
        )
        key = hashlib.sha256(
            source.read_bytes() + version + repr(flags).encode()
        ).hexdigest()
        cache = code_cache_dir() / "swiglu_backward_and_per_token_cast_a5" / key
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "swiglu_backward_and_per_token_cast.so"
        with (cache / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not binary.is_file():
                with tempfile.TemporaryDirectory(dir=cache) as tmp:
                    built = Path(tmp) / binary.name
                    subprocess.run(
                        [
                            str(cc),
                            *flags,
                            str(source),
                            "-o",
                            str(built),
                            "-lascendcl",
                            "-lruntime",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    os.replace(built, binary)
        fn = ctypes.CDLL(str(binary)).launch_fast
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.warn(
            f"Atlas A5 swiglu_backward_and_per_token_cast compiler/library unavailable; using Triton: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 2
    fn.restype = ctypes.c_int
    return fn


def launch(tensors, dimensions):
    import torch

    fn = _native_launcher()
    if fn is None:
        return False
    device = tensors[0].device
    with torch.npu.device(device):
        stream = torch.npu.current_stream(device).npu_stream
        status = fn(stream, *(t.data_ptr() for t in tensors), *dimensions)
    if status:
        raise RuntimeError(f"Ascend C launch failed (ACL status {status})")
    return True
