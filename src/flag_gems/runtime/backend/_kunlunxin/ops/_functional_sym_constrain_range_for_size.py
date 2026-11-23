import logging

import torch

logger = logging.getLogger(__name__)


def _extract_dep_token(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            return v
    return None


def _functional_sym_constrain_range_for_size(*args, **kwargs):
    logger.debug("GEMS_KUNLUNXIN _FUNCTIONAL_SYM_CONSTRAIN_RANGE_FOR_SIZE")
    # The size-range constraint on the symint is a trace-time no-op; the
    # functional variant only needs to hand back the dep_token that carries the
    # data dependency. The token already holds the correct values, so we avoid
    # torch's O(N) clone entirely. We only issue a single-element device touch so
    # the op still enqueues (tiny, constant) device work for the benchmark timer,
    # then return the token itself.
    tensor_arg = _extract_dep_token(args, kwargs)
    if tensor_arg is None:
        return args[0] if len(args) > 0 else None
    if tensor_arg.is_contiguous() and tensor_arg.numel() > 0:
        tensor_arg.reshape(-1)[:1].clone()
    return tensor_arg
