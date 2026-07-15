import threading
from collections import OrderedDict

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes

# Elementwise type promotion is a pure function of a small, cheap descriptor of
# the operands, so pointwise_dynamic recomputes the same result on every launch.
# Cache it to skip the torch._prims_common machinery on the hot path.
#
# What the result depends on (verified against elementwise_dtypes): the promotion
# kind, each tensor operand's dtype and whether it is 0-dim (torch treats a 0-dim
# tensor like a scalar), each python scalar operand's category (bool / int /
# float / complex, never its value), and torch.get_default_dtype() -- a floating
# or complex result can fall back to the default dtype, and INT_TO_FLOAT converts
# integer results using it, so the default dtype is always part of the key. It
# does NOT depend on device, values, non-scalar shape, strides, layout,
# requires_grad, or grad mode.

_TENSOR = "T"
_SCALAR = "S"
# Only these exact python scalar types are canonicalized; anything else falls
# back to the uncached path so we never risk a wrong or unbounded key.
_SCALAR_TYPES = (bool, int, float, complex)


def _operand_descriptors(args):
    """Return a hashable tuple describing the operands, or None if any operand
    is outside the canonicalized fast-path domain (tensor / plain python scalar
    / None)."""
    descriptors = []
    for a in args:
        if isinstance(a, torch.Tensor):
            descriptors.append((_TENSOR, a.dtype, a.ndim == 0))
        elif a is None:
            # torch ignores None operands in promotion.
            continue
        elif type(a) in _SCALAR_TYPES:
            # bool is a subclass of int, so key on the exact type; the value
            # never affects promotion.
            descriptors.append((_SCALAR, type(a)))
        else:
            return None
    return tuple(descriptors)


# Bounded LRU cache from (kind, default_dtype, operand descriptors) to the
# promotion result, guarded by a lock so it stays correct under threaded use.
_promotion_cache = OrderedDict()
_promotion_cache_lock = threading.Lock()
_PROMOTION_CACHE_MAXSIZE = 1024


def type_promotion(*args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    descriptors = _operand_descriptors(args)
    if descriptors is None:
        # An operand outside the supported fast-path domain; compute directly.
        return elementwise_dtypes(*args, type_promotion_kind=type_promotion)

    default_dtype = torch.get_default_dtype()
    key = (type_promotion, default_dtype, descriptors)
    with _promotion_cache_lock:
        cached = _promotion_cache.get(key)
        if cached is not None:
            _promotion_cache.move_to_end(key)  # mark most-recently-used
            return cached

    result = elementwise_dtypes(*args, type_promotion_kind=type_promotion)

    # Only cache if the default dtype did not change while we were computing:
    # otherwise `result` may reflect a different default than `key` records, and
    # storing it would poison the entry for the original default. (This closes a
    # race with a concurrent torch.set_default_dtype.)
    if torch.get_default_dtype() == default_dtype:
        with _promotion_cache_lock:
            _promotion_cache[key] = result
            _promotion_cache.move_to_end(key)
            if len(_promotion_cache) > _PROMOTION_CACHE_MAXSIZE:
                _promotion_cache.popitem(last=False)  # evict least-recently-used
    return result


_accumulator_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}


def get_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    return _accumulator_dtype_map.get(dtype, dtype)
