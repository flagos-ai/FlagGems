import logging

import torch

# C extensions
try:
    from flag_gems import ext_ops  # noqa: F401

    has_c_extension = True
except ImportError:
    has_c_extension = False

from . import testing  # noqa: F401
from . import runtime
from .fused import *  # noqa: F403
from .logging_utils import setup_flaggems_logging
from .ops import *  # noqa: F403
from .patches import *  # noqa: F403
from .runtime.commom_utils import Autograd
from .runtime.register import Register

__version__ = "2.2"
device = runtime.device.name
vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")
registrar = Register
current_work_registrar = None
runtime.replace_customized_ops(globals())


def enable(
    lib=aten_lib,
    unused=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
    forward_only=False,
):
    global current_work_registrar
    current_work_registrar = registrar(
        (
            ("abs", abs, Autograd.disable),
            ("abs_", abs_, Autograd.disable),
            ("add.Tensor", add, Autograd.disable),
            ("add_.Tensor", add_, Autograd.disable),
            ("addmm", addmm, Autograd.disable),
            ("angle", angle, Autograd.disable),
            ("arange.start_step", arange_start, Autograd.disable),
            ("arange.start", arange_start, Autograd.disable),
            ("arange", arange, Autograd.disable),
            ("native_batch_norm", batch_norm, Autograd.disable),
            ("native_batch_norm_backward", batch_norm_backward, Autograd.disable),
            ("bitwise_and.Tensor", bitwise_and_tensor, Autograd.disable),
            ("bitwise_and_.Tensor", bitwise_and_tensor_, Autograd.disable),
            ("bitwise_and.Scalar", bitwise_and_scalar, Autograd.disable),
            ("bitwise_and_.Scalar", bitwise_and_scalar_, Autograd.disable),
            ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, Autograd.disable),
            ("bitwise_not", bitwise_not, Autograd.disable),
            ("bitwise_not_", bitwise_not_, Autograd.disable),
            ("bitwise_or.Tensor", bitwise_or_tensor, Autograd.disable),
            ("bitwise_or_.Tensor", bitwise_or_tensor_, Autograd.disable),
            ("bitwise_or.Scalar", bitwise_or_scalar, Autograd.disable),
            ("bitwise_or_.Scalar", bitwise_or_scalar_, Autograd.disable),
            ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, Autograd.disable),
            ("bmm", bmm, Autograd.disable),
            ("clamp", clamp, Autograd.disable),
            ("clamp_", clamp_, Autograd.disable),
            ("clamp.Tensor", clamp_tensor, Autograd.disable),
            ("clamp_.Tensor", clamp_tensor_, Autograd.disable),
            ("cos", cos, Autograd.disable),
            ("cos_", cos_, Autograd.disable),
            ("pad", pad, Autograd.disable),
            ("constant_pad_nd", constant_pad_nd, Autograd.disable),
            ("cumsum", cumsum, Autograd.disable),
            ("cumsum.out", cumsum_out, Autograd.disable),
            ("cummin", cummin, Autograd.disable),
            ("div.Tensor", true_divide, Autograd.disable),
            ("div_.Tensor", true_divide_, Autograd.disable),
            ("div.Scalar", true_divide, Autograd.disable),
            ("div_.Scalar", true_divide_, Autograd.disable),
            ("div.Tensor_mode", div_mode, Autograd.disable),
            ("div_.Tensor_mode", div_mode_, Autograd.disable),
            ("div.Scalar_mode", div_mode, Autograd.disable),
            ("div_.Scalar_mode", div_mode_, Autograd.disable),
            (
                "divide.Tensor",
                true_divide,
                Autograd.disable,
            ),  # divide, an alias for div
            (
                "divide_.Tensor",
                true_divide_,
                Autograd.disable,
            ),  # divide, an alias for div
            ("divide.Scalar", true_divide, Autograd.disable),
            ("divide_.Scalar", true_divide_, Autograd.disable),
            ("divide.Tensor_mode", div_mode, Autograd.disable),
            ("divide_.Tensor_mode", div_mode_, Autograd.disable),
            ("divide.Scalar_mode", div_mode, Autograd.disable),
            ("divide_.Scalar_mode", div_mode_, Autograd.disable),
            (
                "true_divide.Tensor",
                true_divide,
                Autograd.disable,
            ),  # true_divide, an alias for div
            (
                "true_divide_.Tensor",
                true_divide_,
                Autograd.disable,
            ),  # true_divide, an alias for div
            ("true_divide.Scalar", true_divide, Autograd.disable),
            ("true_divide_.Scalar", true_divide_, Autograd.disable),
            ("floor_divide", floor_divide, Autograd.disable),
            ("floor_divide_.Tensor", floor_divide_, Autograd.disable),
            ("floor_divide.Scalar", floor_divide, Autograd.disable),
            ("floor_divide_.Scalar", floor_divide_, Autograd.disable),
            ("remainder.Tensor", remainder, Autograd.disable),
            ("remainder_.Tensor", remainder_, Autograd.disable),
            ("remainder.Scalar", remainder, Autograd.disable),
            ("remainder_.Scalar", remainder_, Autograd.disable),
            ("remainder.Scalar_Tensor", remainder, Autograd.disable),
            ("native_dropout", dropout, Autograd.disable),
            ("native_dropout_backward", dropout_backward, Autograd.disable),
            ("erf", erf, Autograd.disable),
            ("erf_", erf_, Autograd.disable),
            ("embedding", embedding, Autograd.disable),
            ("embedding_backward", embedding_backward, Autograd.disable),
            ("eq.Tensor", eq, Autograd.disable),
            ("eq.Scalar", eq_scalar, Autograd.disable),
            ("exp", exp, Autograd.disable),
            ("exp_", exp_, Autograd.disable),
            ("exponential_", exponential_, Autograd.disable),
            ("ge.Tensor", ge, Autograd.disable),
            ("ge.Scalar", ge_scalar, Autograd.disable),
            ("gelu", gelu, Autograd.disable),
            ("gelu_", gelu_, Autograd.disable),
            ("gelu_backward", gelu_backward, Autograd.disable),
            ("glu", glu, Autograd.disable),
            ("native_group_norm", group_norm, Autograd.disable),
            ("native_group_norm_backward", group_norm_backward, Autograd.disable),
            ("_weight_norm_interface", weight_norm_interface, Autograd.disable),
            (
                "_weight_norm_interface_backward",
                weight_norm_interface_backward,
                Autograd.disable,
            ),
            ("gt.Tensor", gt, Autograd.disable),
            ("gt.Scalar", gt_scalar, Autograd.disable),
            ("isfinite", isfinite, Autograd.disable),
            ("isin.Tensor_Tensor", isin, Autograd.disable),
            ("isin.Scalar_Tensor", isin, Autograd.disable),
            ("isin.Tensor_Scalar", isin, Autograd.disable),
            ("isinf", isinf, Autograd.disable),
            ("isnan", isnan, Autograd.disable),
            ("minimum", minimum, Autograd.disable),
            ("maximum", maximum, Autograd.disable),
            ("native_layer_norm", layer_norm, Autograd.disable),
            ("native_layer_norm_backward", layer_norm_backward, Autograd.disable),
            ("le.Tensor", le, Autograd.disable),
            ("le.Scalar", le_scalar, Autograd.disable),
            ("lt.Tensor", lt, Autograd.disable),
            ("lt.Scalar", lt_scalar, Autograd.disable),
            ("log", log, Autograd.disable),
            ("rms_norm", rms_norm, Autograd.disable),
            ("rand", rand, Autograd.disable),
            ("randn", randn, Autograd.disable),
            ("rand_like", rand_like, Autograd.disable),
            ("randn_like", randn_like, Autograd.disable),
            ("zeros", zeros, Autograd.disable),
            ("ones", ones, Autograd.disable),
            ("full", full, Autograd.disable),
            ("zeros_like", zeros_like, Autograd.disable),
            ("ones_like", ones_like, Autograd.disable),
            ("full_like", full_like, Autograd.disable),
            ("linspace", linspace, Autograd.disable),
            ("resolve_neg", resolve_neg, Autograd.disable),
            ("resolve_conj", resolve_conj, Autograd.disable),
            ("normal.Tensor_float", normal_tensor_float, Autograd.disable),
            ("normal.float_Tensor", normal_float_tensor, Autograd.disable),
            ("normal.Tensor_Tensor", normal_tensor_tensor, Autograd.disable),
            ("uniform_", uniform_, Autograd.disable),
            ("mean", mean, Autograd.disable),
            ("mean.dim", mean_dim, Autograd.disable),
            ("mm", mm, Autograd.disable),
            ("mul.Tensor", mul, Autograd.disable),
            ("mul_.Tensor", mul_, Autograd.disable),
            ("multinomial", multinomial, Autograd.disable),
            ("mv", mv, Autograd.disable),
            ("nan_to_num", nan_to_num, Autograd.disable),
            ("ne.Tensor", ne, Autograd.disable),
            ("ne.Scalar", ne_scalar, Autograd.disable),
            ("neg", neg, Autograd.disable),
            ("neg_", neg_, Autograd.disable),
            ("pow.Scalar", pow_scalar, Autograd.disable),
            ("pow.Tensor_Scalar", pow_tensor_scalar, Autograd.disable),
            ("pow_.Scalar", pow_tensor_scalar_, Autograd.disable),
            ("pow.Tensor_Tensor", pow_tensor_tensor, Autograd.disable),
            ("pow_.Tensor", pow_tensor_tensor_, Autograd.disable),
            ("reciprocal", reciprocal, Autograd.disable),
            ("reciprocal_", reciprocal_, Autograd.disable),
            ("relu", relu, Autograd.disable),
            ("relu_", relu_, Autograd.disable),
            ("rsqrt", rsqrt, Autograd.disable),
            ("rsqrt_", rsqrt_, Autograd.disable),
            ("sigmoid", sigmoid, Autograd.disable),
            ("sigmoid_", sigmoid_, Autograd.disable),
            ("sigmoid_backward", sigmoid_backward, Autograd.disable),
            ("silu", silu, Autograd.disable),
            ("silu_", silu_, Autograd.disable),
            ("silu_backward", silu_backward, Autograd.disable),
            ("sin", sin, Autograd.disable),
            ("sin_", sin_, Autograd.disable),
            ("_softmax", softmax, Autograd.disable),
            ("_softmax_backward_data", softmax_backward, Autograd.disable),
            ("sort", sort, Autograd.disable),
            ("sub.Tensor", sub, Autograd.disable),
            ("sub_.Tensor", sub_, Autograd.disable),
            ("tanh", tanh, Autograd.disable),
            ("tanh_", tanh_, Autograd.disable),
            ("tanh_backward", tanh_backward, Autograd.disable),
            ("threshold", threshold, Autograd.disable),
            ("threshold_backward", threshold_backward, Autograd.disable),
            ("triu", triu, Autograd.disable),
            ("topk", topk, Autograd.disable),
            ("var_mean.correction", var_mean, Autograd.disable),
            ("linalg_vector_norm", vector_norm, Autograd.disable),
            ("where.self_out", where_self_out, Autograd.disable),
            ("where.self", where_self, Autograd.disable),
            ("where.ScalarSelf", where_scalar_self, Autograd.disable),
            ("where.ScalarOther", where_scalar_other, Autograd.disable),
            ("max", max, Autograd.disable),
            ("max.dim", max_dim, Autograd.disable),
            ("min", min, Autograd.disable),
            ("min.dim", min_dim, Autograd.disable),
            ("amax", amax, Autograd.disable),
            ("argmax", argmax, Autograd.disable),
            ("argmin", argmin, Autograd.disable),
            ("prod", prod, Autograd.disable),
            ("prod.dim_int", prod_dim, Autograd.disable),
            ("sum", sum, Autograd.disable),
            ("sum.dim_IntList", sum_dim, Autograd.disable),
            (
                "scaled_dot_product_attention",
                scaled_dot_product_attention,
                Autograd.disable,
            ),
            ("all", all, Autograd.disable),
            ("all.dim", all_dim, Autograd.disable),
            ("all.dims", all_dims, Autograd.disable),
            ("any", any, Autograd.disable),
            ("any.dim", any_dim, Autograd.disable),
            ("any.dims", any_dims, Autograd.disable),
            ("quantile", quantile, Autograd.disable),
            ("_log_softmax", log_softmax, Autograd.disable),
            ("_log_softmax_backward_data", log_softmax_backward, Autograd.disable),
            ("nll_loss_forward", nll_loss_forward, Autograd.disable),
            ("nll_loss_backward", nll_loss_backward, Autograd.disable),
            ("nll_loss2d_forward", nll_loss2d_forward, Autograd.disable),
            ("nll_loss2d_backward", nll_loss2d_backward, Autograd.disable),
            ("scatter.src", scatter, Autograd.disable),
            ("scatter.reduce", scatter, Autograd.disable),
            ("gather", gather, Autograd.disable),
            ("gather_backward", gather_backward, Autograd.disable),
            ("isclose", isclose, Autograd.disable),
            ("allclose", allclose, Autograd.disable),
            ("fill.Scalar", fill_scalar, Autograd.disable),
            ("fill.Tensor", fill_tensor, Autograd.disable),
            ("fill_.Scalar", fill_scalar_, Autograd.disable),
            ("fill_.Tensor", fill_tensor_, Autograd.disable),
            ("flip", flip, Autograd.disable),
            ("slice_scatter", slice_scatter, Autograd.disable),
            ("select_scatter", select_scatter, Autograd.disable),
            ("index_select", index_select, Autograd.disable),
            ("tile", tile, Autograd.disable),
            ("masked_fill.Tensor", masked_fill, Autograd.disable),
            ("masked_fill.Scalar", masked_fill, Autograd.disable),
            ("masked_fill_.Tensor", masked_fill_, Autograd.disable),
            ("masked_fill_.Scalar", masked_fill_, Autograd.disable),
            ("_unique2", _unique2, Autograd.disable),
            ("_upsample_bicubic2d_aa", _upsample_bicubic2d_aa, Autograd.disable),
            ("upsample_nearest2d", upsample_nearest2d, Autograd.disable),
            ("nonzero", nonzero, Autograd.disable),
            ("repeat", repeat, Autograd.disable),
            ("masked_select", masked_select, Autograd.disable),
            ("stack", stack, Autograd.disable),
            ("hstack", hstack, Autograd.disable),
            ("cat", cat, Autograd.disable),
            (
                "repeat_interleave.self_int",
                repeat_interleave_self_int,
                Autograd.disable,
            ),
            ("vstack", vstack, Autograd.disable),
            ("repeat_interleave.Tensor", repeat_interleave_tensor, Autograd.disable),
            (
                "repeat_interleave.self_Tensor",
                repeat_interleave_self_tensor,
                Autograd.disable,
            ),
            ("randperm", randperm, Autograd.disable),
            ("diag", diag, Autograd.disable),
            ("diag_embed", diag_embed, Autograd.disable),
            ("diagonal_backward", diagonal_backward, Autograd.disable),
            ("index_add", index_add, Autograd.disable),
            ("count_nonzero", count_nonzero, Autograd.disable),
            ("logical_or", logical_or, Autograd.disable),
            ("logical_and", logical_and, Autograd.disable),
            ("polar", polar, Autograd.disable),
            ("logical_xor", logical_xor, Autograd.disable),
            ("logical_not", logical_not, Autograd.disable),
            ("dot", dot, Autograd.disable),
            ("kron", kron, Autograd.disable),
            ("elu", elu, Autograd.disable),
            ("index_put_", index_put_, Autograd.disable),
            ("index_put", index_put, Autograd.disable),
            ("index.Tensor", index, Autograd.disable),
            ("contiguous", contiguous, Autograd.disable),
            ("log_sigmoid", log_sigmoid, Autograd.disable),
            ("vdot", vdot, Autograd.disable),
            ("mse_loss", mse_loss, Autograd.disable),
            ("eye", eye, Autograd.disable),
            ("eye.m", eye_m, Autograd.disable),
            ("to.dtype", to_dtype, Autograd.disable),
        ),
        user_unused_ops_list=[] if unused is None else unused,
        lib=lib,
        forward_only=forward_only,
    )
    setup_flaggems_logging(path=path, record=record, once=once)


class use_gems:
    def __init__(
        self, unused=None, record=False, once=False, path=None, forward_only=False
    ):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused = [] if unused is None else unused
        self.registrar = Register
        self.record = record
        self.once = once
        self.path = path
        self.forward_only = forward_only

    def __enter__(self):
        enable(
            lib=self.lib,
            unused=self.unused,
            registrar=self.registrar,
            record=self.record,
            once=self.once,
            path=self.path,
            forward_only=self.forward_only,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        del current_work_registrar
        del self.forward_only
        if self.record:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO)


def all_ops():
    return current_work_registrar.get_all_ops()


__all__ = [
    "enable",
    "use_gems",
]
