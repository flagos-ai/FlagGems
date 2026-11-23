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

"""Build the small mcFFT ABI shim against the selected, installed MACA SDK.

This module never installs a compiler or SDK. Missing/incompatible development
files are errors. Set FLAGGEMS_CACHE_DIR to the task's cache directory when
running in an isolated validation task.
"""

import ctypes
import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from flag_gems.utils.code_cache import cache_dir

_BUILD_TIMEOUT = 120


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command):
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, timeout=_BUILD_TIMEOUT
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"STFT mcFFT compiler could not run: {exc}") from exc
    if result.returncode:
        diagnostic = result.stderr.decode("utf-8", errors="replace")[-8000:]
        raise RuntimeError(
            f"STFT mcFFT SDK shim compilation failed ({result.returncode}):\n"
            f"{diagnostic}"
        )
    return result


def _sdk_files():
    selected = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME")
    root = Path(selected or "/opt/maca-3.8.1").resolve()
    headers = [root / "include/mcfft/mcfft.h", root / "include/mcfft.h"]
    header = next((path for path in headers if path.is_file()), None)
    if header is None:
        raise RuntimeError(
            f"STFT mcFFT needs the installed MACA header mcfft.h under {root}; "
            "set MACA_PATH to the intended SDK (the static target is MACA 3.8.1)"
        )
    libraries = {
        path.resolve()
        for subdir in ("lib", "lib64")
        for path in (root / subdir).glob("libmcfft.so*")
        if path.is_file()
    }
    if len(libraries) != 1:
        raise RuntimeError(
            f"STFT mcFFT needs exactly one resolved libmcfft.so under {root}; "
            f"found {sorted(str(path) for path in libraries)}"
        )
    library = libraries.pop()
    # Never compile against one SDK and silently execute a preloaded second one.
    maps = Path("/proc/self/maps")
    if maps.is_file():
        loaded = {
            Path(line.split()[-1]).resolve()
            for line in maps.read_text().splitlines()
            if "/libmcfft.so" in line
        }
        if loaded and loaded != {library}:
            raise RuntimeError(
                "STFT mcFFT loaded library does not match the selected MACA SDK: "
                f"selected {library}, loaded {sorted(str(path) for path in loaded)}"
            )
    return root, header, library


def load_mcfft_shim():
    """Return a CDLL exposing the bridge-owned ``gemsMcfft*`` C ABI."""
    if platform.system() != "Linux":
        raise RuntimeError("STFT mcFFT SDK shim requires Linux")
    root, header, library = _sdk_files()
    source = (
        Path(__file__).resolve().parents[1] / "runtime/backend/_metax/stft_fft_shim.cpp"
    )
    if not source.is_file():
        raise RuntimeError(
            f"STFT mcFFT shim source is missing from the package: {source}"
        )
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if len(compiler) != 1:
        raise RuntimeError(
            "STFT mcFFT CXX must name one compiler executable, without wrappers or flags"
        )
    executable = shutil.which(compiler[0]) if compiler else None
    if executable is None:
        raise RuntimeError("STFT mcFFT requires an existing C++17 compiler; set CXX")
    compiler[0] = str(Path(executable).resolve())
    version = _run(compiler + ["--version"]).stdout.decode("utf-8", errors="replace")
    include_dirs = [
        header.parent,
        root / "include",
        root / "include/mcr",
        root / "include/common",
    ]
    options = ["-std=c++17", "-fPIC", "-O2", "-fvisibility=hidden"]
    includes = [f"-I{path}" for path in include_dirs if path.is_dir()]
    preprocess_command = compiler + options + includes + ["-E", str(source)]
    preprocessed = _run(preprocess_command).stdout
    library_hash = _digest(library)
    identity = {
        "source": str(source),
        "source_sha256": _digest(source),
        "sdk": str(root),
        "header": str(header),
        "library": str(library),
        "library_sha256": library_hash,
        "compiler": compiler,
        "compiler_sha256": _digest(Path(compiler[0])),
        "compiler_version": version,
        "compiler_environment": {
            name: os.environ.get(name, "")
            for name in (
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "LD_LIBRARY_PATH",
            )
        },
        "machine": platform.machine(),
        "preprocess_command": preprocess_command,
        "preprocessed_sha256": hashlib.sha256(preprocessed).hexdigest(),
        "link_options": ["-shared", "-Wl,-z,defs", f"-Wl,-rpath,{library.parent}"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = cache_dir() / "stft_mcfft" / key
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "stft_mcfft.so"
    manifest = directory / "manifest.json"
    with (directory / "build.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        valid = False
        if binary.is_file() and manifest.is_file():
            try:
                metadata = json.loads(manifest.read_text())
                valid = metadata.get("binary_sha256") == _digest(binary)
            except (OSError, ValueError):
                pass
        if not valid:
            with tempfile.TemporaryDirectory(
                prefix="build-", dir=directory
            ) as temporary:
                temporary = Path(temporary)
                translation = temporary / "shim.ii"
                translation.write_bytes(preprocessed)
                candidate = temporary / "stft_mcfft.so"
                # Compile the exact preprocessed header closure used for the key.
                command = (
                    compiler
                    + options
                    + identity["link_options"]
                    + [str(translation), str(library), "-o", str(candidate)]
                )
                result = _run(command)
                if _digest(library) != library_hash:
                    raise RuntimeError(
                        "STFT mcFFT SDK library changed during compilation"
                    )
                metadata = dict(identity)
                metadata.update(
                    command=command,
                    binary_sha256=_digest(candidate),
                    compiler_diagnostics=result.stderr.decode(
                        "utf-8", errors="replace"
                    ),
                )
                candidate_manifest = temporary / "manifest.json"
                candidate_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
                os.replace(candidate, binary)
                os.replace(candidate_manifest, manifest)
        # The exact linked library is loaded first, avoiding loader search drift.
        vendor_library = ctypes.CDLL(str(library))
        shim = ctypes.CDLL(str(binary))
        shim._vendor_library = vendor_library
        return shim
