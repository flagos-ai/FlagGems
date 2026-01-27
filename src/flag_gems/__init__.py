import warnings

import torch
from packaging import version

from flag_gems import testing  # noqa: F401
from flag_gems import runtime
from flag_gems.config import aten_patch_list, resolve_user_setting
from flag_gems.experimental_ops import *  # noqa: F403
from flag_gems.fused import *  # noqa: F403
from flag_gems.logging_utils import setup_flaggems_logging, teardown_flaggems_logging
from flag_gems.modules import *  # noqa: F403
from flag_gems.ops import *  # noqa: F403
from flag_gems.patches import *  # noqa: F403
from flag_gems.runtime.register import Register

__version__ = "4.2.1.rc.0"
device = runtime.device.name
vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")
registrar = Register
current_work_registrar = None
runtime.replace_customized_ops(globals())


def torch_ge(v):
    return version.parse(torch.__version__) >= version.parse(v)


_FULL_CONFIG = (
    ("_flash_attention_forward", flash_attention_forward),
    ("_log_softmax", log_softmax),
    ("_log_softmax_backward_data", log_softmax_backward),
    ("_softmax", softmax),
    ("_softmax_backward_data", softmax_backward),
    (
        "_to_copy",
        to_copy,
        lambda: version.parse(torch.__version__) >= version.parse("2.4"),
    ),
    ("_unique2", _unique2),
    ("_upsample_bicubic2d_aa", _upsample_bicubic2d_aa),
    ("_weight_norm_interface", weight_norm_interface),
    ("_weight_norm_interface_backward", weight_norm_interface_backward),
    ("abs", abs),
    ("abs_", abs_),
    ("acos", acos),
    ("add.Tensor", add),
    ("add_.Tensor", add_),
    ("addcdiv", addcdiv),
    ("addcmul", addcmul),
    ("addmv", addmv),
    ("addmv.out", addmv_out),
    ("addmm", addmm),
    ("addmm.out", addmm_out),
    ("addr", addr),
    ("all", all),
    ("all.dim", all_dim),
    ("all.dims", all_dims),
    ("allclose", allclose),
    ("amax", amax),
    ("angle", angle),
    ("any", any),
    ("any.dim", any_dim),
    ("any.dims", any_dims),
    ("arange", arange),
    ("arange.start", arange_start),
    ("arange.start_step", arange_start),
    ("argmax", argmax),
    ("argmin", argmin),
    ("avg_pool2d", avg_pool2d),
    ("avg_pool2d_backward", avg_pool2d_backward),
    ("atan", atan),
    ("atan_", atan_),
    ("baddbmm", baddbmm),
    ("bitwise_and.Scalar", bitwise_and_scalar),
    ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor),
    ("bitwise_and.Tensor", bitwise_and_tensor),
    ("bitwise_and_.Scalar", bitwise_and_scalar_),
    ("bitwise_and_.Tensor", bitwise_and_tensor_),
    ("bitwise_left_shift", bitwise_left_shift),
    ("bitwise_right_shift", bitwise_right_shift),
    ("bitwise_not", bitwise_not),
    ("bitwise_not_", bitwise_not_),
    ("bitwise_or.Scalar", bitwise_or_scalar),
    ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor),
    ("bitwise_or.Tensor", bitwise_or_tensor),
    ("bitwise_or_.Scalar", bitwise_or_scalar_),
    ("bitwise_or_.Tensor", bitwise_or_tensor_),
    ("bmm", bmm),
    ("bmm.out", bmm_out),
    ("cat", cat),
    ("celu", celu),
    ("celu_", celu_),
    ("clamp", clamp),
    ("clamp.Tensor", clamp_tensor),
    ("clamp_min", clamp_min),
    ("clamp_", clamp_),
    ("clamp_.Tensor", clamp_tensor_),
    ("clamp_min_", clamp_min_),
    ("constant_pad_nd", constant_pad_nd),
    # ("contiguous", contiguous),
    ("conv1d", conv1d),
    ("conv1d.padding", conv1d),
    ("conv2d", conv2d),
    ("conv2d.padding", conv2d),
    ("conv3d", conv3d),
    ("conv3d.padding", conv3d),
    (
        "copy_",
        copy_,
        lambda: version.parse(torch.__version__) >= version.parse("2.4"),
    ),
    ("cos", cos),
    ("cos_", cos_),
    ("count_nonzero", count_nonzero),
    ("cummax", cummax),
    ("cummin", cummin),
    ("cumsum", cumsum),
    ("cumsum.out", cumsum_out),
    ("diag", diag),
    ("diag_embed", diag_embed),
    ("diagonal_backward", diagonal_backward),
    ("div.Scalar", true_divide),
    ("div.Scalar_mode", div_mode),
    ("div.Tensor", true_divide),
    ("div.Tensor_mode", div_mode),
    ("div.out", true_divide_out),
    ("div_.Scalar", true_divide_),
    ("div_.Scalar_mode", div_mode_),
    ("div_.Tensor", true_divide_),
    ("div_.Tensor_mode", div_mode_),
    ("divide.Scalar", true_divide),
    ("divide.Scalar_mode", div_mode),
    ("divide.Tensor", true_divide),
    ("divide.Tensor_mode", div_mode),
    ("divide_.Scalar", true_divide_),
    ("divide_.Scalar_mode", div_mode_),
    ("divide_.Tensor", true_divide_),
    ("divide_.Tensor_mode", div_mode_),
    ("dot", dot),
    ("elu", elu),
    ("elu_", elu_),
    ("elu_backward", elu_backward),
    ("embedding", embedding),
    ("embedding_backward", embedding_backward),
    ("eq.Scalar", eq_scalar),
    ("eq.Tensor", eq),
    ("erf", erf),
    ("erf_", erf_),
    ("exp", exp),
    ("exp_", exp_),
    ("exp.out", exp_out),
    ("exp2", exp2),
    ("exp2_", exp2_),
    ("exponential_", exponential_),
    ("eye", eye),
    ("eye.m", eye_m),
    ("fill.Scalar", fill_scalar),
    ("fill.Tensor", fill_tensor),
    ("fill_.Scalar", fill_scalar_),
    ("fill_.Tensor", fill_tensor_),
    ("flip", flip),
    ("floor_divide", floor_divide),
    ("floor_divide.Scalar", floor_divide),
    ("floor_divide_.Scalar", floor_divide_),
    ("floor_divide_.Tensor", floor_divide_),
    ("full", full),
    ("full_like", full_like),
    ("gather", gather),
    ("gather_backward", gather_backward),
    ("ge.Scalar", ge_scalar),
    ("ge.Tensor", ge),
    ("gelu", gelu),
    ("gelu_", gelu_),
    ("gelu_backward", gelu_backward),
    ("glu", glu),
    ("glu_backward", glu_backward),
    ("gt.Scalar", gt_scalar),
    ("gt.Tensor", gt),
    ("hstack", hstack),
    ("index.Tensor", index),
    ("index_add", index_add),
    ("index_add_", index_add_),
    ("index_put", index_put),
    ("index_put_", index_put_),
    ("index_select", index_select),
    ("isclose", isclose),
    ("isfinite", isfinite),
    ("isin.Scalar_Tensor", isin),
    ("isin.Tensor_Scalar", isin),
    ("isin.Tensor_Tensor", isin),
    ("isinf", isinf),
    ("isnan", isnan),
    ("kron", kron),
    ("le.Scalar", le_scalar),
    ("le.Tensor", le),
    ("lerp.Scalar", lerp_scalar),
    ("lerp.Tensor", lerp_tensor),
    ("lerp_.Scalar", lerp_scalar_),
    ("lerp_.Tensor", lerp_tensor_),
    ("linalg_vector_norm", vector_norm),
    ("linspace", linspace),
    ("log", log),
    ("log_sigmoid", log_sigmoid),
    ("logical_and", logical_and),
    ("logical_not", logical_not),
    ("logical_or", logical_or),
    ("logical_xor", logical_xor),
    ("logspace", logspace),
    ("lt.Scalar", lt_scalar),
    ("lt.Tensor", lt),
    ("masked_fill.Scalar", masked_fill),
    ("masked_fill.Tensor", masked_fill),
    ("masked_fill_.Scalar", masked_fill_),
    ("masked_fill_.Tensor", masked_fill_),
    ("masked_scatter", masked_scatter),
    ("masked_scatter_", masked_scatter_),
    ("masked_select", masked_select),
    ("max", max),
    ("max.dim", max_dim),
    ("maximum", maximum),
    ("max_pool2d_with_indices", max_pool2d_with_indices),
    ("max_pool2d_backward", max_pool2d_backward),
    ("mean", mean),
    ("mean.dim", mean_dim),
    ("min", min),
    ("min.dim", min_dim),
    ("minimum", minimum),
    ("mm", mm),
    ("mm.out", mm_out),
    ("moe_sum", moe_sum),
    ("mse_loss", mse_loss),
    ("mul.Tensor", mul),
    ("mul_.Tensor", mul_),
    ("multinomial", multinomial),
    ("mv", mv),
    ("nan_to_num", nan_to_num),
    ("native_batch_norm", batch_norm),
    ("native_batch_norm_backward", batch_norm_backward),
    ("native_dropout", dropout),
    ("native_dropout_backward", dropout_backward),
    ("native_group_norm", group_norm),
    ("native_group_norm_backward", group_norm_backward),
    ("native_layer_norm", layer_norm),
    ("native_layer_norm_backward", layer_norm_backward),
    ("ne.Scalar", ne_scalar),
    ("ne.Tensor", ne),
    ("neg", neg),
    ("neg_", neg_),
    ("nll_loss_backward", nll_loss_backward),
    ("nll_loss_forward", nll_loss_forward),
    ("nll_loss2d_backward", nll_loss2d_backward),
    ("nll_loss2d_forward", nll_loss2d_forward),
    ("nonzero", nonzero),
    ("normal.float_Tensor", normal_float_tensor),
    ("normal.Tensor_float", normal_tensor_float),
    ("normal.Tensor_Tensor", normal_tensor_tensor),
    ("ones", ones),
    ("ones_like", ones_like),
    ("pad", pad),
    ("polar", polar),
    ("pow.Scalar", pow_scalar),
    ("pow.Tensor_Scalar", pow_tensor_scalar),
    ("pow.Tensor_Tensor", pow_tensor_tensor),
    ("pow_.Scalar", pow_tensor_scalar_),
    ("pow_.Tensor", pow_tensor_tensor_),
    ("prod", prod),
    ("prod.dim_int", prod_dim),
    ("quantile", quantile),
    ("rand", rand),
    ("rand_like", rand_like),
    ("randn", randn),
    ("randn_like", randn_like),
    ("randperm", randperm),
    ("reciprocal", reciprocal),
    ("reciprocal_", reciprocal_),
    ("relu", relu),
    ("relu_", relu_),
    ("remainder.Scalar", remainder),
    ("remainder.Scalar_Tensor", remainder),
    ("remainder.Tensor", remainder),
    ("remainder_.Scalar", remainder_),
    ("remainder_.Tensor", remainder_),
    ("repeat", repeat),
    ("repeat_interleave.self_int", repeat_interleave_self_int),
    ("repeat_interleave.self_Tensor", repeat_interleave_self_tensor),
    ("repeat_interleave.Tensor", repeat_interleave_tensor),
    ("resolve_conj", resolve_conj),
    ("resolve_neg", resolve_neg),
    ("rms_norm", rms_norm),
    ("rsqrt", rsqrt),
    ("rsqrt_", rsqrt_),
    ("scaled_softmax_backward", scaled_softmax_backward),
    ("scaled_softmax_forward", scaled_softmax_forward),
    ("scatter.reduce", scatter),
    ("scatter.src", scatter),
    ("scatter_.reduce", scatter_),
    ("scatter_.src", scatter_),
    ("scatter_add_", scatter_add_),
    ("select_scatter", select_scatter),
    ("sigmoid", sigmoid),
    ("sigmoid_", sigmoid_),
    ("sigmoid_backward", sigmoid_backward),
    ("silu", silu),
    ("silu_", silu_),
    ("silu_backward", silu_backward),
    ("sin", sin),
    ("sin_", sin_),
    ("slice_scatter", slice_scatter),
    ("softplus", softplus),
    ("sort", sort),
    ("sort.stable", sort_stable),
    ("sqrt", sqrt),
    ("sqrt_", sqrt_),
    ("stack", stack),
    ("std.correction", std),
    ("sub.Tensor", sub),
    ("sub_.Tensor", sub_),
    ("sum", sum),
    ("sum.dim_IntList", sum_dim),
    ("sum.IntList_out", sum_dim_out),
    ("sum.out", sum_out),
    ("tan", tan),
    ("tan_", tan_),
    ("tanh", tanh),
    ("tanh_", tanh_),
    ("tanh_backward", tanh_backward),
    ("threshold", threshold),
    ("threshold_backward", threshold_backward),
    ("tile", tile),
    ("topk", topk),
    ("trace", trace),
    ("triu", triu),
    ("true_divide.Scalar", true_divide),
    ("true_divide.Tensor", true_divide),
    ("true_divide_.Scalar", true_divide_),
    ("true_divide_.Tensor", true_divide_),
    ("uniform_", uniform_),
    ("upsample_nearest1d", upsample_nearest1d),
    ("upsample_nearest2d", upsample_nearest2d),
    ("var_mean.correction", var_mean),
    ("vdot", vdot),
    ("vstack", vstack),
    ("where.ScalarOther", where_scalar_other),
    ("where.ScalarSelf", where_scalar_self),
    ("where.self", where_self),
    ("where.self_out", where_self_out),
    ("zeros", zeros),
    ("zeros_like", zeros_like),
)


def enable(
    lib=aten_lib,
    unused=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register all FlagGems ops except those explicitly excluded.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        unused: Which ops to skip. Supported forms:
            - list/tuple/set of function names (e.g., ["masked_fill", "mul"]).
            - str path to a YAML file ending with .yml/.yaml containing an
              `exclude:` list.
            - "default" or None: auto-load vendor/arch-specific
              runtime/backend/_<vendor>/[<arch>/]enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable FlagGems logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Notes:
        - If the exclude list/YAML resolves to empty, all ops are registered.
    """
    global current_work_registrar
    exclude_ops = resolve_user_setting(unused, "exclude")
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=[],
        user_exclude_ops=exclude_ops,
        cpp_patched_ops=list(set(aten_patch_list)),
        lib=lib,
    )
    setup_flaggems_logging(path=path, record=record, once=once)


def only_enable(
    lib=aten_lib,
    include=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register only the specified FlagGems ops and skip the rest.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        include: Which ops to register. Supported forms:
            - list/tuple/set of function names (e.g., ["rms_norm", "softmax"]).
            - str path to a YAML file ending with .yml/.yaml (expects a list or
              an `include:` key).
            - "default" or None: auto-load vendor/arch-specific
                runtime/backend/_<vendor>/[<arch>/]only_enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable FlagGems logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Classic usage:
        - Only register a few ops:
            only_enable(include=["rms_norm", "softmax"])
        - Use vendor default YAML:
            only_enable(include="default")  # or include=None
        - Use a custom YAML:
            only_enable(include="/path/to/only_enable.yaml")

    Notes:
        - If the include list/YAML resolves to empty or none of the names match
          known ops, the function warns and returns without registering.
    """
    include_ops = resolve_user_setting(include, "include")
    if not include_ops:
        warnings.warn(
            "only_enable failed: No include entries resolved from list or yaml."
        )
        return

    global current_work_registrar
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=include_ops,
        user_exclude_ops=[],
        cpp_patched_ops=list(set(aten_patch_list)),
        lib=lib,
    )
    setup_flaggems_logging(path=path, record=record, once=once)


class use_gems:
    """
    The 'include' parameter has higher priority than 'exclude'.
    When 'include' is not None, use_gems will not process 'exclude'.
    """

    def __init__(self, exclude=None, include=None, record=False, once=False, path=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.exclude = exclude if isinstance(exclude, (list, tuple, set, str)) else []
        self.include = include if isinstance(include, (list, tuple, set, str)) else []
        self.registrar = Register
        self.record = record
        self.once = once
        self.path = path

    def __enter__(self):
        if self.include:
            only_enable(
                lib=self.lib,
                include=self.include,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )
        else:
            enable(
                lib=self.lib,
                unused=self.exclude,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        if torch.__version__ >= "2.5":
            self.lib._destroy()
        del self.lib
        del self.exclude
        del self.include
        del self.registrar
        del current_work_registrar
        if self.record:
            teardown_flaggems_logging()

    @property
    def experimental_ops(self):
        import flag_gems.experimental_ops

        return flag_gems.experimental_ops


def all_registered_ops():
    return current_work_registrar.get_all_ops()


def all_registered_keys():
    return current_work_registrar.get_all_keys()


__all__ = [
    "enable",
    "only_enable",
    "use_gems",
    "all_registered_ops",
    "all_registered_keys",
]
