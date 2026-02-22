#!/bin/bash
# 运行所有操作的扩展测试

BRANCHES=("codex/cosh" "codex/max_pool3d" "codex/avg_pool3d" "codex/grid_sample" "codex/svd" "codex/ctc_loss")
RESULTS_FILE="/home/qinhaiyan/FlagGems/test_results_summary.txt"

echo "=== FlagGems 扩展测试结果汇总 ===" > $RESULTS_FILE
echo "生成时间: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

for branch in "${BRANCHES[@]}"; do
    echo "=========================================" >> $RESULTS_FILE
    echo "测试分支: $branch" >> $RESULTS_FILE
    echo "=========================================" >> $RESULTS_FILE
    
    cd /home/qinhaiyan/FlagGems
    git checkout $branch 2>&1 | head -2
    
    case $branch in
        "codex/cosh")
            echo "运行 cosh 测试..."
            pytest tests/test_unary_pointwise_ops.py -k cosh -v --tb=short 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/max_pool3d")
            echo "运行 max_pool3d 测试..."
            pytest tests/test_reduction_ops.py -k max_pool3d -v --tb=short 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/avg_pool3d")
            echo "运行 avg_pool3d 测试..."
            pytest tests/test_reduction_ops.py -k avg_pool3d -v --tb=short 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/grid_sample")
            echo "运行 grid_sample 测试..."
            pytest tests/test_special_ops.py -k grid_sample -v --tb=short 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/svd")
            echo "运行 svd 测试..."
            pytest tests/test_special_ops.py -k svd -v --tb=short 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/ctc_loss")
            echo "运行 ctc_loss 测试..."
            pytest tests/test_reduction_ops.py -k ctc_loss -v --tb=short 2>&1 | tee -a $RESULTS_FILE
            ;;
    esac
    
    echo "" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
done

echo "所有测试完成。结果保存在: $RESULTS_FILE"
