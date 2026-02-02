#!/bin/bash

OP_NAME=$1
VERSION=$2

if [ -z "$OP_NAME" ] || [ -z "$VERSION" ]; then
    echo "❌ Usage: bash run_bench.sh <op_name> <version_label>"
    exit 1
fi

DATE=$(date +%Y%m%d)
EXP_DIR="experiments/${OP_NAME}/${DATE}_${VERSION}"
mkdir -p "$EXP_DIR"

# --- 💡 自动识别逻辑开始 ---

echo "🔍 Identifying operator type for: ${OP_NAME}..."

# 1. 自动寻找精度测试文件 (支持 test/ 或 tests/)
ACC_TEST_FILE=$(find tests test -name "*_ops.py" | xargs grep -l "${OP_NAME}" | head -n 1)

# 2. 自动寻找性能测试文件
PERF_TEST_FILE=$(find benchmark -name "*_perf.py" | xargs grep -l "${OP_NAME}" | head -n 1)

if [ -z "$ACC_TEST_FILE" ] || [ -z "$PERF_TEST_FILE" ]; then
    echo "❌ Error: Could not automatically find test files for ${OP_NAME}."
    echo "请确保算子名在 tests/ 和 benchmark/ 对应的脚本中存在。"
    exit 1
fi

echo "📍 Accuracy path: $ACC_TEST_FILE"
echo "📍 Perf path:     $PERF_TEST_FILE"

# --- 🚀 执行流程 ---

echo "🧪 Step 1: Running Accuracy Test..."
pytest "$ACC_TEST_FILE" -k "${OP_NAME}"
if [ $? -ne 0 ]; then
    echo "❌ Accuracy Test Failed!"
    exit 1
fi

echo "🚀 Step 2: Running Performance Benchmark..."
# 使用 --record log 记录结果
pytest "$PERF_TEST_FILE" -k "${OP_NAME}" --record log

# --- 📦 结果归档 ---

LOG_FILE=$(find . -maxdepth 3 -name "result_*.log" -mmin -1 | head -n 1)

if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${EXP_DIR}/result.log"
    echo "✅ Done! Result saved to: ${EXP_DIR}/result.log"
else
    echo "❌ Error: Benchmark log not found."
    exit 1
fi