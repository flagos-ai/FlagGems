#!/bin/bash

# 性能基准测试脚本
# 为各个操作运行基准测试并收集结果

BRANCHES=("codex/cosh" "codex/ctc_loss" "codex/max_pool3d" "codex/grid_sample" "codex/svd" "codex/avg_pool3d")
RESULTS_FILE="/home/qinhaiyan/FlagGems/perf_results_summary.txt"

echo "=== FlagGems Performance Benchmark Results ===" > $RESULTS_FILE
echo "Generated on: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

for branch in "${BRANCHES[@]}"; do
    echo "=========================================" >> $RESULTS_FILE
    echo "Testing branch: $branch" >> $RESULTS_FILE
    echo "=========================================" >> $RESULTS_FILE
    
    cd /home/qinhaiyan/FlagGems
    git checkout $branch
    git pull origin $branch
    
    case $branch in
        "codex/cosh")
            echo "Running cosh benchmark..."
            pytest benchmark/test_unary_pointwise_perf.py -m cosh -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/ctc_loss")
            echo "Running ctc_loss benchmark..."
            pytest benchmark/test_reduction_perf.py -m ctc_loss -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float32 --warmup 50 --iter 100 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/max_pool3d")
            echo "Running max_pool3d benchmark..."
            pytest benchmark/test_reduction_perf.py -m max_pool3d -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/grid_sample")
            echo "Running grid_sample benchmark..."
            pytest benchmark/test_special_perf.py -m grid_sample -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/svd")
            echo "Running svd benchmark..."
            pytest benchmark/test_special_perf.py -m svd -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float32 --warmup 50 --iter 100 2>&1 | tee -a $RESULTS_FILE
            ;;
        "codex/avg_pool3d")
            echo "Running avg_pool3d benchmark..."
            pytest benchmark/test_reduction_perf.py -m avg_pool3d -s --level core --mode kernel --metrics latency_base --metrics latency --metrics speedup --dtypes float16 --dtypes float32 --warmup 50 --iter 100 2>&1 | tee -a $RESULTS_FILE
            ;;
    esac
    
    echo "" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
done

echo "All benchmarks completed. Results saved to: $RESULTS_FILE"
