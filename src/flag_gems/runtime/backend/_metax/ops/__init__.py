from flag_gems.runtime.commom_utils import Autograd

from . import (
    arange,
    exponential_,
    fill,
    full,
    full_like,
    groupnorm,
    isin,
    log_softmax,
    min,
    ones,
    outer,
    prod,
    sigmoid,
    tanh,
    unique,
    zeros,
)


def get_specific_ops():
    return (
        ("arange.start_step", arange.arange_start, Autograd.disable),
        ("arange.start", arange.arange_start, Autograd.disable),
        ("arange", arange.arange, Autograd.disable),
        ("exponential_", exponential_.exponential_, Autograd.disable),
        ("fill.Scalar", fill.fill_scalar, Autograd.disable),
        ("fill.Tensor", fill.fill_tensor, Autograd.disable),
        ("full_like", full_like.full_like, Autograd.disable),
        ("full", full.full, Autograd.disable),
        ("native_group_norm", groupnorm.group_norm, Autograd.enable),
        ("isin.Tensor_Tensor", isin.isin, Autograd.disable),
        ("isin.Scalar_Tensor", isin.isin, Autograd.disable),
        ("isin.Tensor_Scalar", isin.isin, Autograd.disable),
        ("log_softmax.int", log_softmax.log_softmax, Autograd.enable),
        ("min", min.min, Autograd.disable),
        ("min.dim", min.min_dim, Autograd.disable),
        ("ones", ones.ones, Autograd.disable),
        ("prod", prod.prod, Autograd.disable),
        ("prod.dim_int", prod.prod_dim, Autograd.disable),
        ("sigmoid", sigmoid.sigmoid, Autograd.enable),
        ("tanh", tanh.tanh, Autograd.enable),
        ("_unique2", unique._unique2, Autograd.disable),
        ("zeros", zeros.zeros, Autograd.disable),
        ("outer", outer.outer, Autograd.enable),
    )


def get_unused_ops():
    return ()


__all__ = ["get_specific_ops", "get_unused_ops"]
