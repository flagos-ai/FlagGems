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

"""Direct vendor FFT execution for contiguous STFT frames.

No FFT is dispatched through ATen. Both transform directions are unnormalized.
The direct C ABIs are cuFFT (CUDA and CoreX), hipFFT (DTK), and muFFT
(MUSA). The latter two use pointer handles, not cuFFT's int. MetaX uses a
small C ABI shim compiled against its installed SDK to resolve native types.
"""

import atexit
import ctypes
import glob
import os
import sys
import threading
from collections import OrderedDict

import torch

from flag_gems.runtime import device, torch_device_fn

_INT_MAX = (1 << 31) - 1
_MAX_PLANS = 32
_FORWARD = -1
_INVERSE = 1
_R2C, _C2C, _D2Z, _Z2Z = 0x2A, 0x29, 0x6A, 0x69
_CUDA_R_16F, _CUDA_C_16F = 2, 6
_LOCK = threading.RLock()
_PLANS = OrderedDict()
_LIBRARIES = {}
_PID = os.getpid()


def _close_cached_plans():
    # Python exit callbacks run before module teardown, while streams, tensors,
    # and vendor libraries are still alive. A forked child owns no parent plans.
    if os.getpid() != _PID:
        return
    error = None
    with _LOCK:
        for key, plan in list(_PLANS.items()):
            try:
                plan.close()
            except Exception as exc:
                if error is None:
                    error = exc
            else:
                del _PLANS[key]
    if error is not None:
        raise RuntimeError("STFT failed to close cached FFT plans") from error


atexit.register(_close_cached_plans)

# Verified against CUDA/CoreX headers, DTK 26.04 hipfft.h and the MUSA
# muFFT API reference. A same-named function alone is not an ABI guarantee.
_VENDORS = {
    "nvidia": ("cufft", ctypes.c_int, ("libcufft.so.12", "libcufft.so.11")),
    "iluvatar": ("cufft", ctypes.c_int, ("libcufft.so.10",)),
    "hygon": ("hipfft", ctypes.c_void_p, ("libhipfft.so.0",)),
    "mthreads": ("mufft", ctypes.c_void_p, ("libmufft.so.1",)),
    "metax": ("gemsMcfft", ctypes.c_void_p, ()),
}


def _library_candidates(vendor, prefix, sonames):
    # Prefer the exact library already loaded by this process's torch runtime.
    # This avoids selecting a different SDK through an unrelated system path.
    maps = "/proc/self/maps"
    if os.path.isfile(maps):
        with open(maps, encoding="utf-8") as f:
            paths = {line.split()[-1] for line in f if "/" in line}
        for path in sorted(paths):
            if os.path.basename(path).startswith(f"lib{prefix}.so"):
                yield path
    yield from sonames
    yield f"lib{prefix}.so"
    if vendor == "nvidia":
        # pip CUDA 12 uses nvidia/cufft; CUDA 13 uses nvidia/cu13.
        for root in sys.path:
            for package in ("cufft", "cu13"):
                yield from sorted(
                    glob.glob(
                        os.path.join(root, "nvidia", package, "lib/libcufft.so.*")
                    )
                )
    sdk_roots = {
        "nvidia": (os.environ.get("CUDA_HOME"), "/usr/local/cuda"),
        "iluvatar": (os.environ.get("COREX_HOME"), "/usr/local/corex"),
        "hygon": (os.environ.get("ROCM_PATH"), "/opt/dtk"),
        "mthreads": (os.environ.get("MUSA_HOME"), "/usr/local/musa"),
    }
    for root in sdk_roots[vendor]:
        if root:
            for subdir in ("lib", "lib64"):
                yield os.path.join(root, subdir, f"lib{prefix}.so")


class _FFTLibrary:
    def __init__(self, vendor):
        self.prefix, self.handle_type, sonames = _VENDORS[vendor]
        if vendor == "metax":
            from flag_gems.utils.stft_mcfft import load_mcfft_shim

            self.library = load_mcfft_shim()
            # The shim owns these enum codes and maps them using SDK constants.
            self.r2c_type, self.c2c_type = 0, 1
        else:
            self.r2c_type, self.c2c_type = _R2C, _C2C
            failures = []
            for path in dict.fromkeys(
                _library_candidates(vendor, self.prefix, sonames)
            ):
                try:
                    self.library = ctypes.CDLL(path)
                    break
                except OSError as exc:
                    failures.append(str(exc))
            else:
                raise RuntimeError(
                    f"STFT could not load {self.prefix}: " + "; ".join(failures)
                )

        handle = self.handle_type
        integer, pointer = ctypes.c_int, ctypes.c_void_p
        intptr = ctypes.POINTER(integer)
        sizeptr = ctypes.POINTER(ctypes.c_size_t)
        self.create = self._bind("Create", [ctypes.POINTER(handle)])
        self.destroy = self._bind("Destroy", [handle])
        self.set_stream = self._bind("SetStream", [handle, pointer])
        self.set_auto_allocation = self._bind("SetAutoAllocation", [handle, integer])
        self.set_work_area = self._bind("SetWorkArea", [handle, pointer])
        self.make_plan = self._bind(
            "MakePlanMany",
            [
                handle,
                integer,
                intptr,
                intptr,
                integer,
                integer,
                intptr,
                integer,
                integer,
                integer,
                integer,
                sizeptr,
            ],
        )
        self.exec_r2c = self._bind("ExecR2C", [handle, pointer, pointer])
        self.exec_c2c = self._bind("ExecC2C", [handle, pointer, pointer, integer])
        if vendor == "nvidia":
            self.exec_d2z = self._bind("ExecD2Z", [handle, pointer, pointer])
            self.exec_z2z = self._bind("ExecZ2Z", [handle, pointer, pointer, integer])
            longlong = ctypes.c_longlong
            longptr = ctypes.POINTER(longlong)
            self.make_half_plan = self._bind(
                "XtMakePlanMany",
                [
                    handle,
                    integer,
                    longptr,
                    longptr,
                    longlong,
                    longlong,
                    integer,
                    longptr,
                    longlong,
                    longlong,
                    integer,
                    longlong,
                    sizeptr,
                    integer,
                ],
            )
            self.exec_half = self._bind("XtExec", [handle, pointer, pointer, integer])

    def _bind(self, suffix, argtypes):
        name = self.prefix + suffix
        try:
            function = getattr(self.library, name)
        except AttributeError as exc:
            raise RuntimeError(
                f"STFT vendor FFT library does not expose {name}"
            ) from exc
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        return function

    @staticmethod
    def check(function, *args):
        status = function(*args)
        if status:
            raise RuntimeError(f"STFT {function.__name__} failed with status {status}")


class _FFTPlan:
    def __init__(self, library, frames, stream, stream_pointer):
        self.library = library
        self.stream = stream  # Keep the owning stream alive with the plan.
        self.device_index = frames.device.index
        self.handle = library.handle_type()
        self.workspace = None
        library.check(library.create, ctypes.byref(self.handle))
        self._closed = False
        try:
            library.check(library.set_auto_allocation, self.handle, 0)
            batch, n_fft = frames.shape
            complex_input = frames.is_complex()
            output_size = n_fft if complex_input else n_fft // 2 + 1
            work_size = ctypes.c_size_t()
            half = frames.dtype in (torch.float16, torch.complex32)
            if half:
                n = (ctypes.c_longlong * 1)(n_fft)
                output_embed = (ctypes.c_longlong * 1)(output_size)
                input_type = _CUDA_C_16F if complex_input else _CUDA_R_16F
                library.check(
                    library.make_half_plan,
                    self.handle,
                    1,
                    n,
                    n,
                    1,
                    n_fft,
                    input_type,
                    output_embed,
                    1,
                    output_size,
                    _CUDA_C_16F,
                    batch,
                    ctypes.byref(work_size),
                    _CUDA_C_16F,
                )
                self.execute = library.exec_half
                self.has_direction = True
            else:
                double = frames.dtype in (torch.float64, torch.complex128)
                if double:
                    fft_type = _Z2Z if complex_input else _D2Z
                    self.execute = (
                        library.exec_z2z if complex_input else library.exec_d2z
                    )
                else:
                    fft_type = library.c2c_type if complex_input else library.r2c_type
                    self.execute = (
                        library.exec_c2c if complex_input else library.exec_r2c
                    )
                n = (ctypes.c_int * 1)(n_fft)
                output_embed = (ctypes.c_int * 1)(output_size)
                library.check(
                    library.make_plan,
                    self.handle,
                    1,
                    n,
                    n,
                    1,
                    n_fft,
                    output_embed,
                    1,
                    output_size,
                    fft_type,
                    batch,
                    ctypes.byref(work_size),
                )
                self.has_direction = complex_input
            # muFFT rejects SetStream until MakePlanMany initializes the plan.
            library.check(library.set_stream, self.handle, stream_pointer)
            if work_size.value:
                self.workspace = torch.empty(
                    work_size.value, dtype=torch.uint8, device=frames.device
                )
                library.check(
                    library.set_work_area, self.handle, self.workspace.data_ptr()
                )
        except BaseException:
            # Planning may have queued internal initialization. Keep workspace
            # alive until that stream has finished before destroying the plan.
            self.close()
            raise

    def close(self):
        if os.getpid() != _PID:
            return
        with _LOCK:
            if self._closed:
                return
            with torch_device_fn.device(self.device_index):
                self.stream.synchronize()
                self.library.check(self.library.destroy, self.handle)
            self._closed = True
            self.workspace = None

    def run(self, frames, output, stream_pointer, inverse):
        # The global lock protects each mutable plan while ctypes releases the
        # GIL. GPU work is serialized by this plan's single, immutable stream.
        self.library.check(self.library.set_stream, self.handle, stream_pointer)
        frames.record_stream(self.stream)
        output.record_stream(self.stream)
        args = [self.handle, frames.data_ptr(), output.data_ptr()]
        if self.has_direction:
            args.append(_INVERSE if inverse else _FORWARD)
        self.library.check(self.execute, *args)


def fft_frames(frames, *, inverse=False):
    """Transform contiguous ``[batch, n_fft]`` frames without normalization.

    Real input returns the nonredundant half spectrum; complex input returns
    all frequencies. Inverse transforms require complex input. Unsupported
    backends/dtypes raise rather than redispatching or copying to the CPU.
    Plans use 32-bit indexing, with an explicit total-element limit. CUDA
    graph capture is intentionally unsupported, including with warm plans.
    """
    vendor = device.vendor_name
    if vendor == "ascend":
        from flag_gems.runtime.backend._ascend.stft_fft import fft_frames as ascend_fft

        return ascend_fft(frames, inverse=inverse)
    if vendor not in _VENDORS:
        raise NotImplementedError(f"STFT direct vendor FFT is unavailable on {vendor}")
    expected_device = "musa" if vendor == "mthreads" else "cuda"
    if frames.device.type != expected_device:
        raise ValueError(
            f"STFT FFT frames must be on {expected_device}, got {frames.device}"
        )
    if frames.ndim != 2 or not frames.is_contiguous():
        raise ValueError("STFT FFT requires contiguous [batch, n_fft] frames")
    if frames.is_conj() or frames.is_neg():
        raise ValueError(
            "STFT FFT frames must not have unresolved conjugate/negative bits"
        )
    if inverse and not frames.is_complex():
        raise ValueError("STFT inverse FFT requires complex frames")
    accepted = (torch.float32, torch.complex64)
    if vendor == "nvidia":
        accepted += (torch.float64, torch.complex128, torch.float16, torch.complex32)
    if frames.dtype not in accepted:
        raise NotImplementedError(
            f"STFT vendor FFT does not support {frames.dtype} on {vendor}"
        )
    batch, n_fft = frames.shape
    if n_fft < 1 or batch < 1:
        raise ValueError("STFT FFT requires nonempty frames")
    if n_fft * batch > _INT_MAX:
        raise NotImplementedError("STFT vendor FFT requires batch * n_fft <= INT32_MAX")
    if frames.dtype in (torch.float16, torch.complex32) and n_fft & (n_fft - 1):
        raise NotImplementedError(
            "STFT half precision FFT requires a power-of-two n_fft"
        )
    if os.getpid() != _PID:
        raise RuntimeError(
            "STFT vendor FFT cannot be used after fork; use spawn instead"
        )

    with torch_device_fn.device(frames.device.index):
        capturing = getattr(torch_device_fn, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise NotImplementedError("STFT vendor FFT does not support graph capture")
        stream = torch_device_fn.current_stream()
        stream_attribute = "musa_stream" if vendor == "mthreads" else "cuda_stream"
        stream_pointer = getattr(stream, stream_attribute, None)
        if stream_pointer is None:
            raise NotImplementedError(f"STFT FFT requires Stream.{stream_attribute}")
        output_dtype = {
            torch.float16: torch.complex32,
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
        }.get(frames.dtype, frames.dtype)
        output_size = n_fft if frames.is_complex() else n_fft // 2 + 1
        output = torch.empty(
            (batch, output_size), dtype=output_dtype, device=frames.device
        )
        # Include the thread identity for per-thread default stream handles.
        key = (
            vendor,
            frames.device.index,
            frames.dtype,
            n_fft,
            batch,
            tuple(frames.stride()),
            stream_pointer,
            threading.get_ident(),
        )
        with _LOCK:
            plan = _PLANS.get(key)
            if plan is None:
                library = _LIBRARIES.get(vendor)
                if library is None:
                    library = _LIBRARIES[vendor] = _FFTLibrary(vendor)
                if len(_PLANS) >= _MAX_PLANS:
                    old_key, old_plan = next(iter(_PLANS.items()))
                    old_plan.close()
                    del _PLANS[old_key]
                plan = _FFTPlan(library, frames, stream, stream_pointer)
                _PLANS[key] = plan
            _PLANS.move_to_end(key)
            plan.run(frames, output, stream_pointer, inverse)
        return output
