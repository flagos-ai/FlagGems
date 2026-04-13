#!/usr/bin/env python3
"""
FlagGems算子信息提取工具（双维度统计版）

功能：
- 从FlagGems源码提取所有算子信息（两种维度）
- 分析测试文件获取测试详情
- 检测后端支持情况
- 生成多种格式的输出
- **双维度统计**：注册维度 vs 测试覆盖维度

用法：
    # 生成Markdown表格
    python extract_operators.py --output table.md
    python extract_operators.py --format csv --output operators.csv
    python extract_operators.py --format json --output data.json

    # 双维度统计
    python extract_operators.py --dual-stats
    python extract_operators.py --dual-stats --output stats.json
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class FlagGemsOperatorExtractor:
    """FlagGems算子信息提取器"""

    # 所有支持的后端
    ALL_BACKENDS = [
        "NVIDIA",
        "CAMBRICON",
        "METAX",
        "ILUVATAR",
        "MTHREADS",
        "KUNLUNXIN",
        "HYGON",
        "AMD",
        "AIPU",
        "ASCEND",
        "TSINGMICRO",
        "SUNRISE",
        "ENFLAME",
    ]

    # 算子类别映射
    CATEGORY_MAP = {
        # 激活函数
        "relu": "激活函数",
        "relu6": "激活函数",
        "relu_": "激活函数",
        "silu": "激活函数",
        "silu_": "激活函数",
        "silu_backward": "激活函数",
        "gelu": "激活函数",
        "gelu_": "激活函数",
        "gelu_backward": "激活函数",
        "sigmoid": "激活函数",
        "sigmoid_": "激活函数",
        "sigmoid_backward": "激活函数",
        "tanh": "激活函数",
        "tanh_": "激活函数",
        "tanh_backward": "激活函数",
        "elu": "激活函数",
        "elu_": "激活函数",
        "elu_backward": "激活函数",
        "celu": "激活函数",
        "celu_": "激活函数",
        "selu": "激活函数",
        "selu_": "激活函数",
        "softplus": "激活函数",
        "softshrink": "激活函数",
        "hardshrink": "激活函数",
        "hardsigmoid": "激活函数",
        "hardtanh": "激活函数",
        "prelu": "激活函数",
        "rrelu_with_noise_backward": "激活函数",
        "gelu_and_mul": "激活函数",
        "silu_and_mul": "激活函数",
        "geglu": "激活函数",
        "reglu": "激活函数",
        "swiglu": "激活函数",
        # 数学运算
        "abs": "数学运算",
        "abs_": "数学运算",
        "absolute": "数学运算",
        "acos": "数学运算",
        "asin": "数学运算",
        "atan": "数学运算",
        "atan_": "数学运算",
        "cos": "数学运算",
        "cos_": "数学运算",
        "sin": "数学运算",
        "sin_": "数学运算",
        "exp": "数学运算",
        "exp_": "数学运算",
        "exp2": "数学运算",
        "exp2_": "数学运算",
        "log": "数学运算",
        "log1p_": "数学运算",
        "log_sigmoid": "数学运算",
        "sqrt": "数学运算",
        "sqrt_": "数学运算",
        "rsqrt": "数学运算",
        "rsqrt_": "数学运算",
        "pow": "数学运算",
        "pow_": "数学运算",
        "neg": "数学运算",
        "neg_": "数学运算",
        "reciprocal": "数学运算",
        "reciprocal_": "数学运算",
        "erf": "数学运算",
        "erf_": "数学运算",
        "erfinv": "数学运算",
        "i0": "数学运算",
        "i0_": "数学运算",
        "special_i0e": "数学运算",
        "special_i1": "数学运算",
        "hypot": "数学运算",
        "logaddexp": "数学运算",
        "logit": "数学运算",
        "logit_": "数学运算",
        "digamma_": "数学运算",
        "sgn_": "数学运算",
        "polar": "数学运算",
        "arcsinh": "数学运算",
        "arcsinh_": "数学运算",
        "arctanh_": "数学运算",
        "asinh_": "数学运算",
        "tan": "数学运算",
        "tan_": "数学运算",
        "sinh_": "数学运算",
        "angle": "数学运算",
        # 归一化
        "layer_norm": "归一化",
        "native_layer_norm": "归一化",
        "native_layer_norm_backward": "归一化",
        "batch_norm": "归一化",
        "native_batch_norm": "归一化",
        "native_batch_norm_backward": "归一化",
        "group_norm": "归一化",
        "native_group_norm": "归一化",
        "native_group_norm_backward": "归一化",
        "rms_norm": "归一化",
        "instance_norm": "归一化",
        "skip_layernorm": "归一化",
        "fused_add_rms_norm": "归一化",
        "weight_norm": "归一化",
        "weight_norm_interface": "归一化",
        "_weight_norm_interface_backward": "归一化",
        # 注意力机制
        "flash_attention_forward": "注意力机制",
        "_flash_attention_forward": "注意力机制",
        "flash_attn_varlen_func": "注意力机制",
        "scaled_dot_product_attention": "注意力机制",
        "scaled_dot_product_attention_backward": "注意力机制",
        "softmax": "注意力机制",
        "_softmax": "注意力机制",
        "_softmax_backward_data": "注意力机制",
        "safe_softmax": "注意力机制",
        "_safe_softmax": "注意力机制",
        "log_softmax": "注意力机制",
        "_log_softmax": "注意力机制",
        "_log_softmax_backward_data": "注意力机制",
        "scaled_softmax": "注意力机制",
        "scaled_softmax_backward": "注意力机制",
        "scaled_softmax_forward": "注意力机制",
        "flash_mla": "注意力机制",
        "apply_rotary_pos_emb": "注意力机制",
        # 矩阵运算
        "mm": "矩阵运算",
        "bmm": "矩阵运算",
        "addmm": "矩阵运算",
        "addmm_out": "矩阵运算",
        "addmv": "矩阵运算",
        "addmv_out": "矩阵运算",
        "addr": "矩阵运算",
        "matmul": "矩阵运算",
        "mv": "矩阵运算",
        "dot": "矩阵运算",
        "vdot": "矩阵运算",
        "baddbmm": "矩阵运算",
        "baddbmm_backward": "矩阵运算",
        "linear": "矩阵运算",
        "cutlass_scaled_mm": "矩阵运算",
        "w8a8_block_fp8_matmul": "矩阵运算",
        "outer": "矩阵运算",
        # 归约运算
        "sum": "归约运算",
        "mean": "归约运算",
        "max": "归约运算",
        "min": "归约运算",
        "prod": "归约运算",
        "argmax": "归约运算",
        "argmin": "归约运算",
        "cumsum": "归约运算",
        "cummax": "归约运算",
        "cummin": "归约运算",
        "all": "归约运算",
        "any": "归约运算",
        "amax": "归约运算",
        "amin": "归约运算",
        "std": "归约运算",
        "var_mean": "归约运算",
        "vector_norm": "归约运算",
        "linalg_vector_norm": "归约运算",
        "allclose": "归约运算",
        "count_nonzero": "归约运算",
        "nonzero": "归约运算",
        # 卷积池化
        "conv1d": "卷积池化",
        "conv2d": "卷积池化",
        "conv3d": "卷积池化",
        "avg_pool2d": "卷积池化",
        "avg_pool2d_backward": "卷积池化",
        "max_pool2d": "卷积池化",
        "max_pool2d_backward": "卷积池化",
        "max_pool2d_with_indices": "卷积池化",
        "conv_depthwise2d": "卷积池化",
        # 张量操作
        "cat": "张量操作",
        "stack": "张量操作",
        "vstack": "张量操作",
        "hstack": "张量操作",
        "contiguous": "张量操作",
        "alias_copy": "张量操作",
        "to_copy": "张量操作",
        "tril": "张量操作",
        "triu": "张量操作",
        "triu_": "张量操作",
        "diag": "张量操作",
        "diag_embed": "张量操作",
        "diagonal_backward": "张量操作",
        "flip": "张量操作",
        "pixel_unshuffle": "张量操作",
        "slice_backward": "张量操作",
        "slice_scatter": "张量操作",
        "select_scatter": "张量操作",
        "unfold_backward": "张量操作",
        # 索引操作
        "index": "索引操作",
        "gather": "索引操作",
        "gather_backward": "索引操作",
        "scatter": "索引操作",
        "scatter_": "索引操作",
        "scatter_add_": "索引操作",
        "index_select": "索引操作",
        "index_add": "索引操作",
        "index_add_": "索引操作",
        "index_put": "索引操作",
        "index_put_": "索引操作",
        "masked_fill": "索引操作",
        "masked_fill_": "索引操作",
        "masked_select": "索引操作",
        "masked_scatter": "索引操作",
        "masked_scatter_": "索引操作",
        "select_backward": "索引操作",
        "where": "索引操作",
        # 比较运算
        "eq": "比较运算",
        "ne": "比较运算",
        "equal": "比较运算",
        "gt": "比较运算",
        "lt": "比较运算",
        "ge": "比较运算",
        "le": "比较运算",
        "isclose": "比较运算",
        "isfinite": "比较运算",
        "isinf": "比较运算",
        "isnan": "比较运算",
        "isin": "比较运算",
        "maximum": "比较运算",
        "minimum": "比较运算",
        "fmin": "比较运算",
        # 逻辑运算
        "logical_and": "逻辑运算",
        "logical_and_": "逻辑运算",
        "logical_or": "逻辑运算",
        "logical_or_": "逻辑运算",
        "logical_not": "逻辑运算",
        "logical_xor": "逻辑运算",
        # 位运算
        "bitwise_and": "位运算",
        "bitwise_and_": "位运算",
        "bitwise_or": "位运算",
        "bitwise_or_": "位运算",
        "bitwise_not": "位运算",
        "bitwise_not_": "位运算",
        "bitwise_left_shift": "位运算",
        "bitwise_right_shift": "位运算",
        # 算术运算
        "add": "算术运算",
        "add_": "算术运算",
        "sub": "算术运算",
        "sub_": "算术运算",
        "mul": "算术运算",
        "mul_": "算术运算",
        "div": "算术运算",
        "div_": "算术运算",
        "divide": "算术运算",
        "divide_": "算术运算",
        "addcdiv": "算术运算",
        "addcmul": "算术运算",
        "true_divide": "算术运算",
        "true_divide_": "算术运算",
        "floor_divide": "算术运算",
        "floor_divide_": "算术运算",
        "remainder": "算术运算",
        "remainder_": "算术运算",
        "lerp": "算术运算",
        "lerp_": "算术运算",
        "clamp": "算术运算",
        "clamp_": "算术运算",
        "clamp_min": "算术运算",
        "clamp_min_": "算术运算",
        "ceil": "算术运算",
        "ceil_": "算术运算",
        "ceil_out": "算术运算",
        "floor_": "算术运算",
        "zero": "算术运算",
        "zero_": "算术运算",
        # 损失函数
        "cross_entropy_loss": "损失函数",
        "mse_loss": "损失函数",
        "nll_loss": "损失函数",
        "nll_loss2d": "损失函数",
        "nll_loss_nd": "损失函数",
        "nll_loss2d_backward": "损失函数",
        "nll_loss2d_forward": "损失函数",
        "nll_loss_backward": "损失函数",
        "nll_loss_forward": "损失函数",
        "nll_loss_nd_backward": "损失函数",
        "nll_loss_nd_forward": "损失函数",
        "margin_ranking_loss": "损失函数",
        "soft_margin_loss": "损失函数",
        # Dropout
        "dropout": "Dropout",
        "native_dropout": "Dropout",
        "native_dropout_backward": "Dropout",
        # Embedding
        "embedding": "Embedding",
        "embedding_backward": "Embedding",
        "embedding_dense_backward": "Embedding",
        "one_hot": "Embedding",
        # 随机采样
        "rand": "随机采样",
        "rand_like": "随机采样",
        "randn": "随机采样",
        "randn_like": "随机采样",
        "multinomial": "随机采样",
        "randperm": "随机采样",
        "normal": "随机采样",
        "normal_": "随机采样",
        "uniform_": "随机采样",
        "exponential_": "随机采样",
        # 序列生成
        "arange": "序列生成",
        "linspace": "序列生成",
        "logspace": "序列生成",
        # 张量构造
        "zeros": "张量构造",
        "zeros_like": "张量构造",
        "ones": "张量构造",
        "ones_like": "张量构造",
        "full": "张量构造",
        "full_like": "张量构造",
        "eye": "张量构造",
        # 张量填充
        "fill": "张量填充",
        "fill_": "张量填充",
        "constant_pad_nd": "张量填充",
        "pad": "张量填充",
        "reflection_pad1d": "张量填充",
        "reflection_pad2d": "张量填充",
        "replication_pad1d": "张量填充",
        "replication_pad3d": "张量填充",
        # 上下采样
        "upsample_nearest1d": "上下采样",
        "upsample_nearest2d": "上下采样",
        "upsample_nearest3d": "上下采样",
        "upsample_bicubic2d": "上下采样",
        "upsample_bicubic2d_aa": "上下采样",
        "upsample_linear1d": "上下采样",
        "_upsample_nearest_exact1d": "上下采样",
        # 特殊操作
        "unique": "特殊操作",
        "_unique2": "特殊操作",
        "topk": "特殊操作",
        "topk_softmax": "特殊操作",
        "sort": "特殊操作",
        "grouped_topk": "特殊操作",
        "quantile": "特殊操作",
        "trace": "特殊操作",
        "kron": "特殊操作",
        "resolve_conj": "特殊操作",
        "resolve_neg": "特殊操作",
        "conj_physical": "特殊操作",
        "lift_fresh_copy": "特殊操作",
        "bincount": "特殊操作",
        "_functional_sym_constrain_range_for_size": "特殊操作",
        "threshold": "特殊操作",
        "threshold_backward": "特殊操作",
        "t_copy": "特殊操作",
        "repeat": "特殊操作",
        "repeat_interleave": "特殊操作",
        "tile": "特殊操作",
        # MoE
        "fused_moe": "MoE",
        "moe_sum": "MoE",
        "moe_align_block_size": "MoE",
        "per_token_group_quant_fp8": "MoE",
        # RWKV
        "rwkv_ka_fusion": "RWKV",
        "rwkv_mm_sparsity": "RWKV",
    }

    def __init__(self, flaggems_path):
        """
        初始化提取器

        Args:
            flaggems_path: FlagGems项目根目录路径
        """
        self.project_path = Path(flaggems_path)
        self.tests_dir = self.project_path / "tests"
        self.ops_dir = self.project_path / "src" / "flag_gems" / "ops"

        if not self.project_path.exists():
            raise FileNotFoundError(f"FlagGems项目路径不存在: {flaggems_path}")

    def get_category(self, op_name):
        """获取算子类别"""
        if op_name in self.CATEGORY_MAP:
            return self.CATEGORY_MAP[op_name]
        # 模糊匹配
        op_lower = op_name.lower().replace("_", "")
        for key, cat in self.CATEGORY_MAP.items():
            if key.replace("_", "") in op_lower or op_lower in key.replace("_", ""):
                return cat
        return "其他"

    def get_op_file(self, op_name):
        """获取算子源文件"""
        special_map = {
            "layer_norm": "layernorm.py",
            "native_layer_norm": "layernorm.py",
            "native_layer_norm_backward": "layernorm.py",
            "batch_norm": "batch_norm.py",
            "native_batch_norm": "batch_norm.py",
            "group_norm": "groupnorm.py",
            "native_group_norm": "groupnorm.py",
            "rms_norm": "rms_norm.py",
            "instance_norm": "instance_norm.py",
            "flash_attention_forward": "attention.py",
            "flash_attn_varlen_func": "attention.py",
            "scaled_dot_product_attention": "attention.py",
            "softmax": "softmax.py",
            "log_softmax": "softmax.py",
            "safe_softmax": "softmax.py",
        }

        if op_name in special_map:
            return special_map[op_name]

        base_name = (
            op_name.replace("native_", "").replace("_backward", "").replace("_out", "")
        )
        base_name = base_name.lstrip("_")
        return f"{base_name}.py"

    def extract_operators_from_config(self):
        """从_FULL_CONFIG提取所有算子"""
        init_file = self.project_path / "src" / "flag_gems" / "__init__.py"

        if not init_file.exists():
            raise FileNotFoundError(f"找不到__init__.py: {init_file}")

        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()

        operators = []
        for line in content.split("\n"):
            match = re.match(r'\s*\("([^"]+)"', line)
            if match:
                op_full = match.group(1)
                op_base = op_full.split(".")[0]
                operators.append(op_base)

        return sorted(set(operators))

    def analyze_test_files(self):
        """分析所有测试文件"""
        operators = defaultdict(list)
        vendor_skip_stats = defaultdict(set)

        test_files = sorted(self.tests_dir.glob("test_*.py"))

        for test_file in test_files:
            with open(test_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.startswith("@pytest.mark."):
                    marks_block = []
                    parametrize_info = {}
                    skip_vendors = []

                    # 收集标记
                    while i < len(lines) and lines[i].strip().startswith(
                        "@pytest.mark."
                    ):
                        mark_line = lines[i].strip()
                        marks_block.append(mark_line)

                        # 解析skipif
                        if "skipif" in mark_line and "vendor_name" in mark_line:
                            match_eq = re.search(
                                r'vendor_name\s*==\s*"(\w+)"', mark_line
                            )
                            if match_eq:
                                vendor = match_eq.group(1).upper()
                                skip_vendors.append(vendor)
                                vendor_skip_stats[vendor].add(test_file.name)

                        # 解析parametrize
                        if "parametrize" in mark_line:
                            match = re.search(
                                r'parametrize\("([^"]+)",\s*([\w_\[\],\s]+)', mark_line
                            )
                            if match:
                                param_name = match.group(1)
                                param_value = match.group(2).strip()
                                if "[" in param_value:
                                    param_value = param_value.split("[")[0] + "[]"
                                parametrize_info[param_name] = param_value

                        i += 1

                    # 获取test函数名
                    if i < len(lines) and "def test_" in lines[i]:
                        func_match = re.search(r"def (test_\w+)", lines[i])
                        if func_match:
                            test_name = func_match.group(1)

                            # 提取算子标记
                            op_mark = None
                            for mark in marks_block:
                                if mark.startswith("@pytest.mark.") and all(
                                    x not in mark
                                    for x in ["parametrize", "skip", "inplace"]
                                ):
                                    op_mark = mark.replace("@pytest.mark.", "").strip()
                                    if op_mark and len(op_mark) > 1:
                                        break

                            if op_mark:
                                operators[op_mark].append(
                                    {
                                        "test_name": test_name,
                                        "test_file": test_file.name,
                                        "dtypes": parametrize_info.get(
                                            "dtype", "FLOAT_DTYPES"
                                        ),
                                        "shapes": parametrize_info.get(
                                            "shape", "POINTWISE_SHAPES"
                                        ),
                                        "skip_vendors": skip_vendors,
                                        "is_skip": len(skip_vendors) > 0,
                                    }
                                )
                else:
                    i += 1

        return dict(operators), dict(vendor_skip_stats)

    def generate_markdown_table(self, operators, vendor_skip_stats):
        """生成Markdown表格"""
        lines = []

        # 标题
        lines.append("# FlagGems 算子信息表\n")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"> 项目路径: {self.project_path}\n")
        lines.append("\n---\n\n")

        # 统计信息
        all_ops = self.extract_operators_from_config()
        total_tests = sum(len(tests) for tests in operators.values())

        lines.append("## 统计概览\n")
        lines.append(f"- **唯一算子名称**: {len(all_ops)}\n")
        lines.append(f"- **测试标记数**: {len(operators)}\n")
        lines.append(f"- **测试方法数**: {total_tests}\n")
        lines.append("\n---\n\n")

        # 后端支持统计
        lines.append("## 后端支持统计\n")
        lines.append("\n| 后端 | Skip测试数 | 状态 |\n")
        lines.append("|------|-----------|------|\n")

        vendor_skip_count = {v: len(s) for v, s in vendor_skip_stats.items()}
        for vendor in self.ALL_BACKENDS:
            skip_count = vendor_skip_count.get(vendor, 0)
            status = "✅ 支持" if skip_count == 0 else f"⚠️ {skip_count}个skip"
            lines.append(f"| {vendor} | {skip_count} | {status} |\n")

        lines.append("\n---\n\n")

        # 算子详情表格
        lines.append("## 算子详情\n")

        # 按类别分组
        operators_by_category = defaultdict(list)
        for op_name, tests in operators.items():
            cat = self.get_category(op_name)
            operators_by_category[cat].append((op_name, tests))

        # 生成表格
        for category in sorted(operators_by_category.keys()):
            lines.append(f"\n### {category}\n")
            lines.append(
                "\n| 算子名称 | 算子类别 | 算子所在文件 | 算子测例 "
                "| 数据类型 | 形状 | 测试文件 | 正确性测试命令 "
                "| 性能测试命令 | 支持的后端 |\n"
            )
            lines.append(
                "|----------|----------|--------------|----------"
                "|----------|------|----------|----------------"
                "|--------------|----------|\n"
            )

            for op_name, tests in sorted(operators_by_category[category]):
                for test_info in tests:
                    # 后端支持
                    skip_vendors = test_info.get("skip_vendors", [])
                    if len(skip_vendors) == 0:
                        backend_support = "全部后端"
                    else:
                        backend_support = f"全部后端（跳过{'、'.join(skip_vendors)}）"

                    skip_mark = " (skip)" if test_info["is_skip"] else ""
                    test_name = test_info["test_name"] + skip_mark
                    op_file = self.get_op_file(op_name)
                    correct_cmd = f'pytest -m "{op_name}" --ref cpu'
                    perf_cmd = f'pytest -m "{op_name}" --level core --record log'

                    lines.append(
                        f"| {op_name} | {category} | {op_file} "
                        f"| {test_name} | {test_info['dtypes']} "
                        f"| {test_info['shapes']} "
                        f"| {test_info['test_file']} "
                        f"| {correct_cmd} | {perf_cmd} "
                        f"| {backend_support} |\n"
                    )

        return "".join(lines)

    def generate_csv(self, operators, vendor_skip_stats):
        """生成CSV格式"""
        import io

        output = io.StringIO()

        # CSV头
        output.write("算子名称,算子类别,算子所在文件,算子测例,数据类型,形状,测试文件,正确性测试命令,性能测试命令,支持的后端,是否skip\n")

        for op_name, tests in sorted(operators.items()):
            category = self.get_category(op_name)
            op_file = self.get_op_file(op_name)

            for test_info in tests:
                skip_vendors = test_info.get("skip_vendors", [])
                backend_support = (
                    "全部后端" if len(skip_vendors) == 0 else f"跳过{','.join(skip_vendors)}"
                )

                skip_mark = " (skip)" if test_info["is_skip"] else ""
                test_name = test_info["test_name"] + skip_mark
                correct_cmd = f'pytest -m "{op_name}" --ref cpu'
                perf_cmd = f'pytest -m "{op_name}" --level core --record log'
                is_skip = "是" if test_info["is_skip"] else "否"

                output.write(
                    f"{op_name},{category},{op_file},{test_name},"
                    f'{test_info["dtypes"]},{test_info["shapes"]},'
                    f'{test_info["test_file"]},{correct_cmd},'
                    f"{perf_cmd},{backend_support},{is_skip}\n"
                )

        return output.getvalue()

    def generate_json(self, operators, vendor_skip_stats):
        """生成JSON格式"""
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_path": str(self.project_path),
                "unique_operators": len(self.extract_operators_from_config()),
                "test_markers": len(operators),
                "total_tests": sum(len(tests) for tests in operators.values()),
                "backend_skip_stats": {v: len(s) for v, s in vendor_skip_stats.items()},
            },
            "operators": {},
        }

        for op_name, tests in operators.items():
            category = self.get_category(op_name)
            op_file = self.get_op_file(op_name)

            data["operators"][op_name] = {
                "category": category,
                "source_file": op_file,
                "tests": tests,
            }

        return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="FlagGems算子信息提取工具（双维度统计版）")
    parser.add_argument("--path", default=".", help="FlagGems项目路径（默认为当前目录）")
    parser.add_argument(
        "--format",
        choices=["markdown", "csv", "json"],
        default="markdown",
        help="输出格式（默认：markdown）",
    )
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细处理信息")
    parser.add_argument(
        "--dual-stats",
        action="store_true",
        help="输出双维度统计报告（注册维度 + 测试覆盖维度）",
    )

    args = parser.parse_args()

    try:
        # 创建提取器
        extractor = FlagGemsOperatorExtractor(args.path)

        if args.verbose:
            print(f"项目路径: {extractor.project_path}")
            print(f"测试目录: {extractor.tests_dir}")
            print(f"算子目录: {extractor.ops_dir}")

        # 双维度统计模式
        if args.dual_stats:
            return run_dual_stats(extractor, args)

        # 原有功能：生成表格
        if not args.output:
            print("❌ 错误: 需要指定 --output 参数")
            return 1

        if args.verbose:
            print("\n正在分析测试文件...")

        operators, vendor_skip_stats = extractor.analyze_test_files()

        if args.verbose:
            print(f"找到 {len(operators)} 个算子标记")
            print(f"总共 {sum(len(tests) for tests in operators.values())} 个测试方法")
            print(f"有skip标记的后端: {list(vendor_skip_stats.keys())}")

        # 生成输出
        if args.format == "markdown":
            content = extractor.generate_markdown_table(operators, vendor_skip_stats)
        elif args.format == "csv":
            content = extractor.generate_csv(operators, vendor_skip_stats)
        else:  # json
            content = extractor.generate_json(operators, vendor_skip_stats)

        # 写入文件
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ 成功生成算子信息表: {args.output}")
        print(f"   格式: {args.format}")
        print(f"   算子数: {len(operators)}")

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_dual_stats(extractor, args):
    """运行双维度统计"""
    # 维度1: 注册维度 - 从 _FULL_CONFIG 提取
    registered_ops = set(extractor.extract_operators_from_config())

    # 维度2: 测试覆盖维度 - 从 pytest marks 提取
    tested_ops, _ = extractor.analyze_test_files()
    tested_ops_set = set(tested_ops.keys())

    # 差异分析
    both = registered_ops & tested_ops_set
    registered_only = registered_ops - tested_ops_set
    tested_only = tested_ops_set - registered_ops

    # 类别统计
    category_count = defaultdict(int)
    for op in tested_ops_set:
        cat = extractor.get_category(op)
        category_count[cat] += 1

    # 输出结果
    print("=" * 70)
    print("FlagGems 算子统计报告（双维度）")
    print(f"生成时间: {datetime.now().isoformat()}")
    print("=" * 70)

    print("\n【维度1: 注册维度】从 _FULL_CONFIG 提取")
    print(f"  - 注册的唯一算子数: {len(registered_ops)}个")

    print("\n【维度2: 测试覆盖维度】从 pytest marks 提取")
    print(f"  - 有测试覆盖的算子数: {len(tested_ops_set)}个")

    print("\n【差异分析】")
    print(f"  - 两个维度都有的算子: {len(both)}个")
    print(f"  - 仅在注册中有（无测试）: {len(registered_only)}个")
    print(f"  - 仅在测试中有（未注册）: {len(tested_only)}个")

    print("\n【测试覆盖类别分布】")
    for cat, count in sorted(category_count.items(), key=lambda x: -x[1]):
        print(f"  - {cat}: {count}个")

    # 保存JSON
    if args.output:
        stats_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "repo_root": str(extractor.project_path),
            },
            "registered_dimension": {
                "unique_count": len(registered_ops),
                "operators": sorted(registered_ops),
            },
            "tested_dimension": {
                "count": len(tested_ops_set),
                "operators": sorted(tested_ops_set),
                "category_distribution": dict(
                    sorted(category_count.items(), key=lambda x: -x[1])
                ),
            },
            "diff_analysis": {
                "both_count": len(both),
                "registered_only_count": len(registered_only),
                "tested_only_count": len(tested_only),
                "registered_only": sorted(registered_only),
                "tested_only": sorted(tested_only),
            },
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 统计结果已保存到: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
