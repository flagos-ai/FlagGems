import argparse
import logging

import pytest

# This is a collection of unit test by op name for testing the
# accuracy of each op.

blas_ops_ut_map = {
    "linear": ("test_accuracy_addmm",),
    "bmm": ("test_accuracy_bmm",),
    "mv": ("test_accuracy_mv",),
    "test_accuracy_mm": ("test_accuracy_mm",),
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
    "groupnorm": ("test_accuracy_groupnorm",),
    "native_groupnorm": ("test_accuracy_groupnorm",),
    "layernorm": ("test_accuracy_layernorm",),
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
    "sub": (
        "test_accuracy_sub",
        "test_accuracy_sub_tensor_scalar",
        "test_accuracy_sub_scalar_tensor",
    ),
}


special_ops_ut_map = {
    "dropout": ("test_accuracy_dropout",),
    "native_dropout": ("test_accuracy_dropout",),
}

op_name_to_unit_test_maps = {
    "test_blas_ops.py": blas_ops_ut_map,
    "test_reduction_ops.py": reduction_ops_ut_map,
    "test_unary_pointwise_ops.py": unary_pointwise_ops_ut_map,
    "test_binary_pointwise_ops.py": binary_pointwise_ops_ut_map,
    "test_special_ops.py": special_ops_ut_map,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="test for all ops in the op list"
    )
    parser.add_argument("--name", type=str, help="test for a specific op")
    args = parser.parse_args()

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
                    result = pytest.main(["-s", cmd])
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
                        result = pytest.main(["-s", cmd])
                        final_result += result
        print("final_result: ", final_result)
        exit(final_result)
