from backend_utils import Autograd

from . import add, gelu


def get_specific_ops():
    return (
        ("add.Tensor", add.add, Autograd.disable),
        ("gelu", gelu.gelu, Autograd.enable),
    )


def get_unused_ops():
    return ("cumsum", "cos")


__all__ = ["get_specific_ops", "get_unused_ops"]
