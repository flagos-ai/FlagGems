#!/bin/bash

# 1. 参数检查与设置
VENDOR=${1:?"Usage: bash tools/run_backend_tests_thead.sh <vendor>"}
export GEMS_VENDOR=$VENDOR

# 2. 显卡设备设置
# 平头哥 PPU 通常使用 PPU_VISIBLE_DEVICES 环境变量
# 根据之前的日志，设备 ID 为 0, 1, 2, 3。这里默认使用 0 号卡进行测试。
export PPU_VISIBLE_DEVICES=0

echo "Running FlagGems tests on T-Head PPU with GEMS_VENDOR=$GEMS_VENDOR"
echo "Target Device: PPU $PPU_VISIBLE_DEVICES"

# 3. 环境激活
# 使用 pyenv 管理 Python 版本
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"

# 4. 依赖安装
echo "Setting up Python environment..."
pip install -U pip
pip install uv

# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装构建工具
uv pip install setuptools==82.0.1 scikit-build-core==0.12.2 pybind11==3.0.3 cmake==3.31.10 ninja==1.13.0

# 安装项目依赖
# 如果平头哥有特定的 extra 依赖组（例如 .[tpu] 或 .[custom]），请在此处修改。
uv pip install -e .[test]

# 如果需要安装平头哥特定的运行时库（例如 torch_t头哥版），请在这里添加
# uv pip install torch-ppu ...

# 5. 加载测试工具
source tools/run_command.sh

echo "Starting tests..."

# 6. 执行测试用例

# Reduction ops
run_command pytest -s tests/test_reduction_ops.py
run_command pytest -s tests/test_general_reduction_ops.py
run_command pytest -s tests/test_norm_ops.py

# Pointwise ops
run_command pytest -s tests/test_pointwise_dynamic.py
run_command pytest -s tests/test_unary_pointwise_ops.py
run_command pytest -s tests/test_binary_pointwise_ops.py
run_command pytest -s tests/test_pointwise_type_promotion.py
run_command pytest -s tests/test_tensor_constructor_ops.py

# BLAS ops
# 注意：如果平头哥不支持某些特定的 Attention 机制，可能需要跳过此测试
run_command pytest -s tests/test_attention_ops.py
run_command pytest -s tests/test_blas_ops.py

# Special ops
run_command pytest -s tests/test_special_ops.py
run_command pytest -s tests/test_distribution_ops.py

# Convolution ops
run_command pytest -s tests/test_convolution_ops.py

# Utils
run_command pytest -s tests/test_libentry.py
run_command pytest -s tests/test_shape_utils.py
run_command pytest -s tests/test_tensor_wrapper.py

# Examples
# 保持注释状态，因为网络可能不可达
# run_command pytest -s examples/model_bert_test.py

echo "All tests finished."
