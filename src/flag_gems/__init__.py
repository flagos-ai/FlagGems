import logging
import os
def _get_framework():
    framework_env = os.environ.get("FLAGGEMS_FRAMEWORK", "paddle").lower()
    
    if framework_env == "paddle":
        try:
            import paddle
            return paddle, "paddle"
        except ImportError:
            print("Warning: Paddle not found, falling back to torch")
            framework_env = "torch"
    
    if framework_env == "torch":
        try:
            import torch
            return torch, "torch"
        except ImportError:
            raise ImportError("Neither torch nor paddle is available")
    
    raise ValueError(f"Unsupported framework: {framework_env}")

dl_framework, framework_name = _get_framework()

if framework_name == "paddle":
    import paddle
    paddle.compat.enable_torch_proxy()


from flag_gems import testing  # noqa: F401
from flag_gems import runtime
from flag_gems.config import aten_patch_list
# from flag_gems.fused import *  # noqa: F403
from flag_gems.logging_utils import setup_flaggems_logging
# from flag_gems.modules import *  # noqa: F403
from flag_gems.ops import *  # noqa: F403
# from flag_gems.patches import *  # noqa: F403
from flag_gems.runtime.register import Register
from flag_gems.runtime.register_paddle import RegisterPaddle

setup_flaggems_logging(record=True, once=False)
__version__ = "3.0"
device = runtime.device.name
vendor_name = runtime.device.vendor_name

if framework_name == "torch":
    aten_lib = dl_framework.library.Library("aten", "IMPL")
    registrar = Register
elif framework_name == "paddle":
    aten_lib = None
    registrar = RegisterPaddle

current_work_registrar = None
runtime.replace_customized_ops(globals())

def enable(
    lib=aten_lib,
    unused=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    global current_work_registrar

    user_unused_ops_list = list(set(unused or []))
    cpp_patched_ops_list = list(set(aten_patch_list))

    if framework_name == "torch":
        op_config = (
            ("_flash_attention_forward", flash_attention_forward),
            ("_log_softmax", log_softmax),
            ("_log_softmax_backward_data", log_softmax_backward),
            ("_softmax", softmax),
            ("_softmax_backward_data", softmax_backward),
            ("_unique2", _unique2),
            ("_upsample_bicubic2d_aa", _upsample_bicubic2d_aa),
            ("_weight_norm_interface", weight_norm_interface),
            ("_weight_norm_interface_backward", weight_norm_interface_backward),
            ("abs", abs),
            ("abs_", abs_),
            ("add.Tensor", add),
            ("add_.Tensor", add_),
            ("addcdiv", addcdiv),
            ("addmv", addmv),
            ("addmv.out", addmv_out),
            ("addmm", addmm),
            ("addmm.out", addmm_out),
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
            ("atan", atan),
            ("atan_", atan_),
            ("bitwise_and.Scalar", bitwise_and_scalar),
            ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor),
            ("bitwise_and.Tensor", bitwise_and_tensor),
            ("bitwise_and_.Scalar", bitwise_and_scalar_),
            ("bitwise_and_.Tensor", bitwise_and_tensor_),
            ("bitwise_not", bitwise_not),
            ("bitwise_not_", bitwise_not_),
            ("bitwise_or.Scalar", bitwise_or_scalar),
            ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor),
            ("bitwise_or.Tensor", bitwise_or_tensor),
            ("bitwise_or_.Scalar", bitwise_or_scalar_),
            ("bitwise_or_.Tensor", bitwise_or_tensor_),
            ("bmm", bmm),
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
            ("contiguous", contiguous),
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
            # ("index.Tensor", index),
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
            ("masked_select", masked_select),
            ("max", max),
            ("max.dim", max_dim),
            ("maximum", maximum),
            ("mean", mean),
            ("mean.dim", mean_dim),
            ("min", min),
            ("min.dim", min_dim),
            ("minimum", minimum),
            ("mm", mm),
            ("mm.out", mm_out),
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
            ("addcmul", addcmul),
            ("softplus", softplus),
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
            ("sqrt", sqrt),
            ("sqrt_", sqrt_),
            ("rsqrt", rsqrt),
            ("rsqrt_", rsqrt_),
            ("scatter.reduce", scatter),
            ("scatter.src", scatter),
            ("scatter_.reduce", scatter_),
            ("scatter_.src", scatter_),
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
            ("sort", sort),
            ("sort.stable", sort_stable),
            ("stack", stack),
            ("sub.Tensor", sub),
            ("sub_.Tensor", sub_),
            ("sum", sum),
            ("sum.dim_IntList", sum_dim),
            ("sum.IntList_out", sum_dim_out),
            ("sum.out", sum_out),
            ("tanh", tanh),
            ("tanh_", tanh_),
            ("tanh_backward", tanh_backward),
            ("threshold", threshold),
            ("threshold_backward", threshold_backward),
            ("tile", tile),
            ("to.dtype", to_dtype),
            ("topk", topk),
            ("triu", triu),
            ("true_divide.Scalar", true_divide),
            ("true_divide.Tensor", true_divide),
            ("true_divide_.Scalar", true_divide_),
            ("true_divide_.Tensor", true_divide_),
            ("uniform_", uniform_),
            ("upsample_nearest2d", upsample_nearest2d),
            ("var_mean.correction", var_mean),
            ("vdot", vdot),
            ("addr", addr),
            ("vstack", vstack),
            ("where.ScalarOther", where_scalar_other),
            ("where.ScalarSelf", where_scalar_self),
            ("where.self", where_self),
            ("where.self_out", where_self_out),
            ("zeros", zeros),
            ("zeros_like", zeros_like),
        )
        current_work_registrar = registrar(
            op_config,
            user_unused_ops_list=user_unused_ops_list,
            cpp_patched_ops_list=cpp_patched_ops_list,
            lib=lib,
        )
    elif framework_name == "paddle":
        op_config = (# When replacing _C_ops.func, it is necessary to maintain consistency between the custom operator and the input parameters of _C_ops.func. 
        #When directly replacing the function of paddle.xxx, after flag_gems.enable, it needs to be re added to the function of paddle.xxx
            ("_C_ops.softmax", softmax),
            ("_C_ops.bmm", bmm), # Todo: fix oom
            ("_C_ops.sum", sum),
            ("_C_ops.mean", mean),
            ("_C_ops.triu", triu),
            ("_C_ops.addmm",addmm),
            ("_C_ops.all", all),
            ("_C_ops.amax", amax),
            ("_C_ops.argmax", argmax),
            ("_C_ops.diag", diag),
            ("_C_ops.dot", dot), 
            ("_C_ops.embedding", embedding), 
            ("_C_ops.ones", ones),
            ("_C_ops.zeros", zeros),
            ("_C_ops.zeros_like", zeros_like),
            ("_C_ops.topk", topk),
            ("_C_ops.max", max),
            ("_C_ops.min", min),
            ("_C_ops.mv", mv),
            ("_C_ops.index_select", index_select)
        )
        current_work_registrar = registrar(
            op_config,
            user_unused_ops_list=user_unused_ops_list
        )
    setup_flaggems_logging(path=path, record=record, once=once)


class use_gems:
    def __init__(self, unused=None, record=False, once=False, path=None):
        if framework_name == "torch":
            self.lib = dl_framework.library.Library("aten", "IMPL")
        else:
            self.lib = None  
        
        self.unused = [] if unused is None else unused
        self.registrar = registrar
        self.record = record
        self.once = once
        self.path = path

    def __enter__(self):
        enable(
            lib=self.lib,
            unused=self.unused,
            registrar=self.registrar,
            record=self.record,
            once=self.once,
            path=self.path,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        if framework_name == "paddle":
            current_work_registrar.restore_all()
        del current_work_registrar
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
