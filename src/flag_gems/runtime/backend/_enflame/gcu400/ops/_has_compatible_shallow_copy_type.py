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

logger = logging.getLogger(__name__)

# aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
#
# This is a metadata-only operator: it reports whether the TensorImpl of
# ``self`` can shallow-copy the TensorImpl type of ``from``. It performs no
# element-wise computation, so there is no Triton kernel involved.
#
# The predicate is defined on DispatchKeySets, not on ``Tensor.layout``. From
# c10::TensorImpl::has_compatible_shallow_copy_type, two TensorImpls are
# compatible when their key sets are equal, or when both are dense, both are
# sparse (COO), or both are sparse compressed:
#
#     return (key_set_ == from) || (is_dense(key_set_) && is_dense(from)) ||
#         (is_sparse(key_set_) && is_sparse(from)) ||
#         (is_sparse_compressed(key_set_) && is_sparse_compressed(from));
#
# where ``is_dense``/``is_sparse`` additionally require the backend component to
# be one of an explicit allow-list. That backend requirement is why ``layout``
# alone is not sufficient: a meta tensor carries the Dense functionality key but
# the Meta backend bit, so it is *not* dense-compatible and only matches another
# tensor with an identical key set. Nested, MKL-DNN and quantized tensors also
# report ``torch.strided`` while living on their own backends.
#
# On this platform the GCU backend registers through the PrivateUse1 dispatch
# key (torch_gcu), and the native fallback kernel for this op treats a dense
# PrivateUse1 key set as a supported dense backend: e.g. a dense CPU tensor and
# a dense GCU tensor are shallow-copy compatible (verified against the native
# op across CPU/GCU/meta/nested/quantized/sparse pairs). The stock allow-list
# below therefore carries PrivateUse1, listed first because GCU is the local
# hot backend and the family scan is a linear ``has()`` probe.

# Backend components accepted by each family in TensorImpl.h.
_DENSE_BACKENDS = ("PrivateUse1", "CPU", "CUDA", "MPS", "HIP", "XPU", "HPU", "MTIA")
_SPARSE_BACKENDS = ("CPU", "CUDA", "MPS", "HIP", "XPU")

# ``DispatchKeySet.has`` takes a *runtime* key, i.e. a functionality key already
# combined with a backend bit ("SparseCPU"), so the allow-lists are expressed as
# those combined names rather than as bare backend components -- the Python
# bindings do not expose BackendComponent bits on their own.
_DENSE_RUNTIME_KEYS = _DENSE_BACKENDS
_SPARSE_RUNTIME_KEYS = tuple("Sparse" + b for b in _SPARSE_BACKENDS)
_SPARSE_CSR_RUNTIME_KEYS = tuple("SparseCsr" + b for b in _SPARSE_BACKENDS)

# Module-level bindings keep the hot path free of repeated attribute lookups.
_dispatch_keys = torch._C._dispatch_keys


def _parse_keys(names):
    """Resolve dispatch key names, ignoring any absent from this build."""
    keys = []
    for name in names:
        try:
            keys.append(torch._C._dispatch_key_parse(name))
        except (RuntimeError, AttributeError):
            continue
    return tuple(keys)


_DENSE_KEYS = _parse_keys(_DENSE_RUNTIME_KEYS)
_SPARSE_KEYS = _parse_keys(_SPARSE_RUNTIME_KEYS)
_SPARSE_CSR_KEYS = _parse_keys(_SPARSE_CSR_RUNTIME_KEYS)


def _has_any(key_set, keys):
    for key in keys:
        if key_set.has(key):
            return True
    return False


def _is_dense(key_set):
    return _has_any(key_set, _DENSE_KEYS)


def _is_sparse(key_set):
    return _has_any(key_set, _SPARSE_KEYS)


def _is_sparse_compressed(key_set):
    return _has_any(key_set, _SPARSE_CSR_KEYS)


def _has_compatible_shallow_copy_type(self: torch.Tensor, from_: torch.Tensor) -> bool:
    """Return True if ``self`` can shallow-copy the TensorImpl type of ``from_``.

    Mirrors ``c10::TensorImpl::has_compatible_shallow_copy_type``.

    Args:
        self: The destination tensor.
        from_: The source tensor whose TensorImpl type is checked.

    Returns:
        bool: True when the two TensorImpls are shallow-copy compatible.
    """
    logger.debug("GEMS_ENFLAME _HAS_COMPATIBLE_SHALLOW_COPY_TYPE")
    self_keys = _dispatch_keys(self)
    from_keys = _dispatch_keys(from_)

    # pybind DispatchKeySet equality compares the raw key-set bits, which is
    # cheaper than building and comparing the two raw_repr() strings and is
    # exactly the ``key_set_ == from`` clause of the native predicate.
    if self_keys == from_keys:
        return True
    if _is_dense(self_keys) and _is_dense(from_keys):
        return True
    if _is_sparse(self_keys) and _is_sparse(from_keys):
        return True
    return _is_sparse_compressed(self_keys) and _is_sparse_compressed(from_keys)
