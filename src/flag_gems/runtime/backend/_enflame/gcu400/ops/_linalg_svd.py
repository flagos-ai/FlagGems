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

import logging

import torch

# gcu400: bind the enflame-local svd (device-type gates, GCU kernel fixes).
# Absolute import per the linalg_matmul/expand_copy backend precedent; the
# generic flag_gems.ops.svd copy stays untouched as the semantic baseline.
from flag_gems.runtime.backend._enflame.gcu400.ops.svd import svd

logger = logging.getLogger(__name__)

# Drivers the native CUDA op accepts. Measured against `aten::_linalg_svd` on
# CUDA: "gesvd" and "gesvda" are accepted and "gesdd" is not (the CPU backend
# rejects every driver at dispatch time, so the CUDA set is the relevant one).
# The strings are case-sensitive on native.
_VALID_DRIVERS = ("gesvd", "gesvda")


def _unknown_svd_driver_error(driver):
    return RuntimeError(
        f"torch.linalg.svd: unknown svd driver {driver} in svd_cusolver "
        "computation. Check doc at "
        "https://pytorch.org/docs/stable/generated/torch.linalg.svd.html"
    )


def _linalg_svd(A, full_matrices=False, compute_uv=True, *, driver=None):
    """Triton implementation of ``aten::_linalg_svd``.

    Delegates to the hand-written Triton SVD kernels in ``svd.py``.

    ``aten::_linalg_svd`` is the private primitive that ``torch.linalg.svd`` /
    ``torch.linalg.svdvals`` decompose into. It returns ``(U, S, Vh)`` such that
    ``A = U diag(S) Vh``. The internal ``svd`` helper follows the ``torch.svd``
    convention and returns ``(U, S, V)`` (with ``A = U diag(S) V^H``), so we
    transpose-conjugate ``V`` to obtain ``Vh``. ``full_matrices=True``
    corresponds to ``some=False`` (full U/V matrices) in the ``torch.svd``
    convention.

    When ``compute_uv`` is ``False``, ``aten::_linalg_svd`` returns empty ``U``
    and ``Vh`` tensors (only ``S`` is materialized), unlike the ``torch.svd``
    convention which returns zero-filled factors; we honor the aten contract
    here.
    """
    logger.debug("GEMS _LINALG_SVD")
    # ``driver`` is validated before anything else, matching the CUDA native op,
    # which rejects an unknown driver regardless of the input dtype. The Triton
    # kernels have no driver concept -- they always compute the reduced SVD and
    # expand to the full factors, which is what both cuSOLVER drivers do from
    # the caller's point of view -- so a valid driver only has to be accepted,
    # not acted on.
    if driver is not None and driver not in _VALID_DRIVERS:
        raise _unknown_svd_driver_error(driver)
    # The Triton SVD kernels only implement float32; higher/lower precision
    # matrices (e.g. float16/bfloat16) are not supported by the backend.
    if A.dtype != torch.float32:
        raise TypeError(f"_linalg_svd only supports float32 input, got {A.dtype}")
    U, S, V = svd(A, some=not full_matrices, compute_uv=compute_uv)
    if not compute_uv:
        # aten::_linalg_svd leaves U/Vh empty when compute_uv is False.
        empty = A.new_empty(0)
        return empty, S, empty
    Vh = V.mH
    return U, S, Vh


# torch_gcu's native ``aten::_linalg_svd`` rejects every ``driver=`` value with
# a CUDA-only guard ("keyword argument `driver=` is only supported on CUDA
# inputs with cuSOLVER backend"), so the CUDA driver contract -- gesvd/gesvda
# accepted, anything else rejected with "unknown svd driver" -- is unreachable
# on GCU at the aten surface. ``torch.linalg.svd``/``svdvals`` are C++ bindings
# and stay native, but direct Python packet calls (``torch.ops.aten._linalg_svd``)
# can be given the CUDA contract by shadowing the namespace attribute: validate
# the driver like cuSOLVER, strip a valid one (the Triton/native kernels have no
# driver concept), and forward everything else to the original packet untouched.
# Same install-at-import shim pattern as the complex ``torch.Tensor.to`` /
# scalar-mul shims in svd.py.
#
# The shim must remain a genuine ``torch._ops.OpOverloadPacket`` (subclass), not
# a plain object: ``torch._inductor.lowering`` does ``make_fallback(aten._linalg_svd)``
# at *module import* time (triggered lazily by the first ``torch.compile`` call,
# e.g. the module-level decorator in ``vllm.utils.deep_gemm``), and a plain
# shadow object makes that raise
# ``RuntimeError: Unsupported fallback <shim> with type <class ...>``, which
# crashes any first-time inductor import in the process. Subclassing keeps
# ``isinstance``/``overloads()``/overload resolution identical to the real
# packet, so inductor (and dynamo) treat the shim exactly like the native
# packet; only ``__call__`` gains the driver contract.
class _AtenLinalgSvdDriverShim(torch._ops.OpOverloadPacket):
    def __init__(self, packet):
        super().__init__(
            packet._qualified_op_name,
            packet.__name__,
            packet._op,
            packet._overload_names,
        )

    def __call__(self, *args, **kwargs):
        driver = kwargs.get("driver")
        if driver is not None and driver not in _VALID_DRIVERS:
            raise _unknown_svd_driver_error(driver)
        if driver is not None:
            kwargs = dict(kwargs, driver=None)
        return super().__call__(*args, **kwargs)


def _install_aten_svd_driver_shim():
    aten_ns = torch.ops.aten
    current = getattr(aten_ns, "_linalg_svd", None)
    if current is None or getattr(current, "_flaggems_gcu_svd_driver_shim", False):
        return
    shim = _AtenLinalgSvdDriverShim(current)
    shim._flaggems_gcu_svd_driver_shim = True
    aten_ns._linalg_svd = shim


_install_aten_svd_driver_shim()
