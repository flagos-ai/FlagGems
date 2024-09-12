import argparse
import logging

import pytest

# This is a collection of unit test by op name for testing the
# accuracy of each op.

blas_ops_ut_map = {
    "linear": ("test_accuracy_addmm",),
    "addmm": ("test_accuracy_addmm",),
    "bmm": ("test_accuracy_bmm",),
    "mv": ("test_accuracy_mv",),
    "mm": ("test_accuracy_mm",),
    "matmul": ("test_accuracy_mm",),
    "outer": ("test_accuracy_outer",),
}

reduction_ops_ut_map = {
    "all": (
        "test_accuracy_all",
        "test_accuracy_all_dim",
        "test_accuracy_all_dims",
    ),
    "cross_entropy_loss": (
        "test_accuracy_cross_entropy_loss_indices",
        "test_accuracy_cross_entropy_loss_probabilities",
    ),
    "group_norm": ("test_accuracy_groupnorm",),
    "native_group_norm": ("test_accuracy_groupnorm",),
    "layer_norm": ("test_accuracy_layernorm",),
    "native_layer_norm": ("test_accuracy_layernorm",),
    "log_softmax": ("test_accuracy_log_softmax",),
    "max": (
        "test_accuracy_max",
        "test_accuracy_max_dim",
    ),
    "mean": (
        "test_accuracy_mean",
        "test_accuracy_mean_dim",
    ),
    "min": (
        "test_accuracy_min",
        "test_accuracy_min_dim",
    ),
    "prod": (
        "test_accuracy_prod",
        "test_accuracy_prod_dim",
    ),
    "softmax": ("test_accuracy_softmax",),
    "sum": (
        "test_accuracy_sum",
        "test_accuracy_sum_dim",
    ),
    "var_mean": ("test_accuracy_varmean",),
    "amax": ("test_accuracy_amax",),
    "any": (
        "test_accuracy_any",
        "test_accuracy_any_dim",
        "test_accuracy_any_dims",
    ),
    "argmax": ("test_accuracy_argmax",),
    "cumsum": ("test_accuracy_cumsum",),
    "rmsnorm": ("test_accuracy_rmsnorm",),
    "skip_layernorm": ("test_accuracy_skip_layernorm",),
    "skip_rmsnorm": ("test_accuracy_skip_rmsnorm",),
    "vector_norm": ("test_accuracy_vectornorm",),
}

unary_pointwise_ops_ut_map = {
    "abs": ("test_accuracy_abs",),
    "bitwise_not": ("test_accuracy_bitwisenot",),
    "cos": ("test_accuracy_cos",),
    "exp": ("test_accuracy_exp",),
    "gelu": ("test_accuracy_gelu",),
    "isinf": ("test_accuracy_isinf",),
    "isnan": ("test_accuracy_isnan",),
    "neg": ("test_accuracy_neg",),
    "reciprocal": ("test_accuracy_reciprocal",),
    "relu": ("test_accuracy_relu",),
    "rsqrt": ("test_accuracy_rsqrt",),
    "sigmoid": ("test_accuracy_sigmoid",),
    "silu": ("test_accuracy_silu",),
    "sin": ("test_accuracy_sin",),
    "tanh": ("test_accuracy_tanh",),
    "triu": ("test_accuracy_triu",),
    "erf": ("test_accuracy_erf",),
    "isfinite": ("test_accuracy_isfinite",),
    "flip": ("test_accuracy_flip", "test_accuracy_flip_with_non_dense_input"),
}

distribution_ops_ut_map = {
    "normal": ("test_accuracy_normal",),
    "uniform": ("test_accuracy_uniform",),
    "exponential_": ("test_accuracy_exponential_",),
}

tensor_constructor_ops_ut_map = {
    "rand": ("test_accuracy_rand",),
    "randn": ("test_accuracy_randn",),
    "rand_like": ("test_accuracy_rand_like",),
    "zeros": ("test_accuracy_zeros",),
    "zeros_like": ("test_accuracy_zeros_like",),
    "ones": ("test_accuracy_ones",),
    "ones_like": ("test_accuracy_ones_like",),
    "full": ("test_accuracy_full",),
    "full_like": ("test_accuracy_full_like",),
}

binary_pointwise_ops_ut_map = {
    "add": (
        "test_accuracy_add",
        "test_accuracy_add_tensor_scalar",
        "test_accuracy_add_scalar_tensor",
    ),
    "bitwise_and": (
        "test_accuracy_bitwiseand",
        "test_accuracy_bitwiseand_scalar",
        "test_accuracy_bitwiseand_scalar_tensor",
    ),
    "bitwise_or": (
        "test_accuracy_bitwiseor",
        "test_accuracy_bitwiseor_scalar",
        "test_accuracy_bitwiseor_scalar_tensor",
    ),
    "or": (
        "test_accuracy_bitwiseor",
        "test_accuracy_bitwiseor_scalar",
        "test_accuracy_bitwiseor_scalar_tensor",
    ),
    "div": (
        "test_accuracy_div",
        "test_accuracy_div_tensor_scalar",
        "test_accuracy_div_scalar_tensor",
    ),
    "eq": ("test_accuracy_eq", "test_accuracy_eq_scalar"),
    "ge": ("test_accuracy_ge", "test_accuracy_ge_scalar"),
    "gt": ("test_accuracy_gt", "test_accuracy_gt_scalar"),
    "le": ("test_accuracy_le", "test_accuracy_le_scalar"),
    "lt": ("test_accuracy_lt", "test_accuracy_lt_scalar"),
    "mul": (
        "test_accuracy_mul",
        "test_accuracy_mul_tensor_scalar",
        "test_accuracy_mul_scalar_tensor",
    ),
    "ne": ("test_accuracy_ne", "test_accuracy_ne_scalar"),
    "pow": (
        "test_accuracy_pow",
        "test_accuracy_pow_scalar_tensor",
        "test_accuracy_pow_tensor_scalar",
    ),
    "rsub": ("test_accuracy_rsub",),
    "clamp": ("test_accuracy_clamp", "test_accuracy_clamp_tensor"),
    "gelu_and_mul": ("test_accuracy_gelu_and_mul",),
    "silu_and_mul": ("test_accuracy_silu_and_mul",),
    "where": (
        "test_accuracy_where_self",
        "test_accuracy_where_scalar_self",
        "test_accuracy_where_scalar_other",
    ),
    "isclose": ("test_accuracy_isclose",),
    "allclose": ("test_accuracy_allclose",),
    "sub": (
        "test_accuracy_sub",
        "test_accuracy_sub_tensor_scalar",
        "test_accuracy_sub_scalar_tensor",
    ),
}

special_ops_ut_map = {
    "dropout": ("test_accuracy_dropout",),
    "native_dropout": ("test_accuracy_dropout",),
    "apply_rotary_position_embedding": ("test_apply_rotary_pos_emb",),
    "embedding": ("test_embedding",),
    "resolve_neg": ("test_accuracy_resolve_neg",),
    "resolve_conj": ("test_accuracy_resolve_conj",),
}

op_name_to_unit_test_maps = {
    "test_blas_ops.py": blas_ops_ut_map,
    "test_reduction_ops.py": reduction_ops_ut_map,
    "test_unary_pointwise_ops.py": unary_pointwise_ops_ut_map,
    "test_distribution_ops.py": distribution_ops_ut_map,
    "test_tensor_constructor_ops.py": tensor_constructor_ops_ut_map,
    "test_binary_pointwise_ops.py": binary_pointwise_ops_ut_map,
    "test_special_ops.py": special_ops_ut_map,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="test for all ops in the op list"
    )
    parser.add_argument("--name", type=str, help="test for a specific op")
    parser.add_argument(
        "--device",
        action="store",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to run reference tests on. Choose 'cuda' or 'cpu'. Default is 'cuda'.",
    )
    args = parser.parse_args()

    device = args.device
    print(f"Running tests on device: {device}...")

    op_nums = 0
    op_list = []
    for item in op_name_to_unit_test_maps.values():
        op_nums = op_nums + len(item)
        for op in item.keys():
            op_list.append(op)
    print(f"Here is the sorted op list with {op_nums} ops:")
    op_list = sorted(op_list)
    print(op_list)

    final_result = 0
    if args.all:
        for file_name, collection in op_name_to_unit_test_maps.items():
            for op, uts in collection.items():
                for ut in uts:
                    cmd = f"{file_name}::{ut}"
                    result = pytest.main(["-s", cmd, "--device", device])
        print("final_result: ", final_result)
        exit(final_result)

    if args.name:
        if args.name not in op_list:
            logging.fatal(f"No op named {args.name} found! Check the name and list!")
            exit(1)

        for file_name, collection in op_name_to_unit_test_maps.items():
            for op, uts in collection.items():
                if op == args.name:
                    print(op)
                    for ut in uts:
                        cmd = f"{file_name}::{ut}"
                        print(cmd)
                        result = pytest.main(["-s", cmd, "--device", device])
                        final_result += result
        print("final_result: ", final_result)
        exit(final_result)
