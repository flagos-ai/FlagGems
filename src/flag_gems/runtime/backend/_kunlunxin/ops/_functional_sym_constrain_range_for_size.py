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
    tensor_arg = _extract_dep_token(args, kwargs)
    if tensor_arg is None:
        return args[0] if len(args) > 0 else None
    if tensor_arg.is_contiguous() and tensor_arg.numel() > 0:
        tensor_arg.reshape(-1)[:1].clone()
    return tensor_arg
