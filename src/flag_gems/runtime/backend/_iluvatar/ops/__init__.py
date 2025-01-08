from backend_utils import Autograd

from . import bmm, mm


def get_specific_ops():
    return (
        ("mm", mm.mm, Autograd.disable),
        ("bmm", bmm.bmm, Autograd.disable),
    )


def get_unused_ops():
    # The following operations are marked as unused because the
    # triton.language.core.reshape function is not supported in Triton 2.1.
    return ("randperm", "topk", "sort", "multinomial")


__all__ = ["get_specific_ops", "get_unused_ops"]
