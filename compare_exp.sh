#!/bin/bash

# 使用方法: 
# 1. 对比指定两个版本: bash compare_exp.sh log10 v1_inplace baseline
# 2. 默认对比输出到报告文件

OP_NAME=$1
NEW_VER=$2
BASE_VER=$3

if [ -z "$OP_NAME" ] || [ -z "$NEW_VER" ] || [ -z "$BASE_VER" ]; then
    echo "❌ 用法: bash compare_exp.sh <算子名> <新版本> <基准版本>"
    echo "例子: bash compare_exp.sh log10 v1_inplace baseline"
    exit 1
fi

# 自动寻找对应的 result.log 路径 (匹配最新日期的文件夹)
NEW_LOG=$(ls -t experiments/${OP_NAME}/*_${NEW_VER}/result.log 2>/dev/null | head -n 1)
BASE_LOG=$(ls -t experiments/${OP_NAME}/*_${BASE_VER}/result.log 2>/dev/null | head -n 1)

if [ -z "$NEW_LOG" ] || [ -z "$BASE_LOG" ]; then
    echo "❌ 找不到指定的日志文件，请检查版本号是否正确。"
    exit 1
fi

REPORT_FILE="experiments/summary_reports/${OP_NAME}_${NEW_VER}_vs_${BASE_VER}.txt"

echo "📊 正在对比分析..."
echo "🆕 新版本: $NEW_LOG"
echo "基准版: $BASE_LOG"

# 执行对比脚本
python benchmark/summary_for_plot.py "$NEW_LOG" --compare "$BASE_LOG" > "$REPORT_FILE"

# 同时在终端打印出结果，方便直接查看
cat "$REPORT_FILE"

echo -e "\n✅ 报告已生成: $REPORT_FILE"