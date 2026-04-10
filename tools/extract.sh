#!/bin/bash
# FlagGems算子信息提取脚本便捷脚本
# 使用方法:
#   ./extract.sh [format]          # 生成表格 (markdown/csv/json)
#   ./extract.sh --dual-stats      # 双维度统计
#   ./extract.sh --dual-stats json # 双维度统计并保存JSON

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查是否为双维度统计模式
if [ "$1" = "--dual-stats" ]; then
    echo "====================================="
    echo "FlagGems 算子双维度统计"
    echo "====================================="
    echo ""
    echo "项目路径: $PROJECT_ROOT"
    
    OUTPUT_ARG=""
    if [ -n "$2" ] && [ "$2" = "json" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        OUTPUT_DIR="$PROJECT_ROOT/docs/generated"
        OUTPUT_FILE="$OUTPUT_DIR/dual_stats_$TIMESTAMP.json"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_ARG="--output $OUTPUT_FILE"
    fi
    
    python3 "$SCRIPT_DIR/extract_operators.py" \
        --path "$PROJECT_ROOT" \
        --dual-stats \
        $OUTPUT_ARG
    
    if [ -n "$OUTPUT_ARG" ]; then
        echo ""
        echo "查看结果：cat $OUTPUT_FILE"
    fi
    exit 0
fi

FORMAT=${1:-markdown}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/docs/generated"
OUTPUT_FILE="$OUTPUT_DIR/operators_$TIMESTAMP.$FORMAT"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "====================================="
echo "FlagGems 算子信息提取工具"
echo "====================================="
echo ""
echo "项目路径: $PROJECT_ROOT"
echo "输出格式: $FORMAT"
echo "输出路径: $OUTPUT_FILE"
echo ""

# 执行提取
python3 "$SCRIPT_DIR/extract_operators.py" \
    --path "$PROJECT_ROOT" \
    --format "$FORMAT" \
    --output "$OUTPUT_FILE" \
    --verbose

echo ""
echo "✅ 提取完成！"
echo ""
echo "查看结果："
echo "  cat $OUTPUT_FILE"
echo ""

# 如果是markdown，创建一个latest链接
if [ "$FORMAT" = "markdown" ]; then
    LATEST="$OUTPUT_DIR/operators_latest.md"
    cp "$OUTPUT_FILE" "$LATEST"
    echo "最新版本："
    echo "  $LATEST"
fi