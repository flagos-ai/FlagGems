"""
Metax (沐曦) backend specialization for the Cast operator.

Naming convention — why ``to.py`` instead of ``cast.py``:
-------------------------------------------------------------------------
In PyTorch, the user-level dtype-cast operation (``tensor.to(dtype=...)``)
is dispatched through the ATen operator ``aten::_to_copy``, **not** an
operator named ``cast``.  Therefore every FlagGems backend implements its
Cast specialization under the name ``to_copy`` in a file called ``to.py``:

===============  ============================
Backend            File
===============  ============================
_cambricon         ``_cambricon/ops/to.py``
_enflame           ``_enflame/gcu300/ops/to.py``
_kunlunxin         ``_kunlunxin/ops/to.py``
_metax (this one)  ``_metax/ops/to.py``
===============  ============================

The ``to_copy`` function is exported through :file:`__init__.py`'s
``__all__`` list and then registered automatically by FlagGems'
``Register`` class, which overrides the default ``aten::_to_copy`` on the
Metax dispatch key.

Verification:
-------------------------------------------------------------------------
All test cases passed on Metax C550 GPU:

+-----------------------------+------+
| Test case                   | Pass |
+-----------------------------+------+
| float32 → float16           | ✓    |
| float16 → float32           | ✓    |
| float32 → bfloat16          | ✓    |
| bfloat16 → float32          | ✓    |
| int32 → float32             | ✓    |
| float32 → int32             | ✓    |
| int64 → int32               | ✓    |
| non-contiguous stride keep  | ✓    |
| 1024×1024 matrix cast       | ✓    |
+-----------------------------+------+
"""

import logging
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger("flag_gems." + __name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
    return x


def _resolve_dtype(x: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return x.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(f"Unsupported dtype argument type: {type(dtype)!r}")


def _resolve_device(x: torch.Tensor, device: Optional[torch.device]) -> torch.device:
    if device is None:
        return x.device
    return torch.device(device)


def _normalize_memory_format(
    memory_format: Optional[torch.memory_format],
) -> torch.memory_format:
    if memory_format is None:
        return torch.preserve_format
    return memory_format


def _allocate_preserve_format(x: torch.Tensor, empty_kwargs: dict) -> torch.Tensor:
    """Recreate tensor storage while honoring preserve_format semantics."""
    if torch.ops.aten.is_non_overlapping_and_dense(x):
        return torch.empty_strided(x.size(), x.stride(), **empty_kwargs)
    # Fall back to PyTorch's best-effort layout suggestion when stride replication is unsafe.
    return torch.empty_like(x, memory_format=torch.preserve_format, **empty_kwargs)


# func: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#   bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    # We only implement the dense strided kernel today; all other layouts fall back to PyTorch.
    if (layout is not None and layout != torch.strided) or x.layout != torch.strided:
        raise NotImplementedError(
            "FlagGems to_copy currently supports strided tensors only."
        )
    if pin_memory is not None:
        raise NotImplementedError(
            "FlagGems to_copy does not yet support pin_memory=True."
        )
    if x.is_quantized:
        raise NotImplementedError(
            "Quantized tensors are not supported in FlagGems to_copy yet."
        )

    target_dtype = _resolve_dtype(x, dtype)
    target_device = _resolve_device(x, device)
    target_memory_format = _normalize_memory_format(memory_format)

    # Triton does not support complex dtypes; fall back to PyTorch.
    if x.dtype.is_complex or target_dtype.is_complex:
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    if target_device != x.device or (
        x.device.type == "cpu" and target_device.type == "cpu"
    ):
        # Device transfer (d2h/h2d etc.) relies on PyTorch's implementation.
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    logger.debug("METAX GEMS _TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    return _to_copy_func(x, out0=out)
