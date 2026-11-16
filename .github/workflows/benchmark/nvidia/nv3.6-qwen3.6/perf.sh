#!/bin/bash

# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# Usage & Description (使用方法与说明)
# ==============================================================================
# 功能:
#     执行性能压测的控制脚本。会自动锁定 GPU 频率，读取 config.env 中的
#     测试场景配置，循环调用 all_perf.py 发起压测，并在结束时自动恢复 GPU 频率。
#
# 前置条件:
#     1. 确保 vLLM 服务端已通过 start.sh 成功启动并处于就绪状态。
#     2. 当前目录下必须存在 config.env 配置文件。
#     3. 需提前通过环境变量指定要压测的 GPU（如 CUDA_VISIBLE_DEVICES）。
#
# 使用方法:
#     # 方式一：默认执行（读取 config.env 中的 BENCH_SCENARIOS 运行多组测试）
#     bash perf.sh
#
#     # 方式二：在特定的 GPU 上运行
#     export CUDA_VISIBLE_DEVICES=0,1
#     bash perf.sh
#
#     # 方式三：临时覆盖某项配置（如临时修改总请求数）
#     BENCH_NUM_PROMPTS=1000 bash perf.sh
#
# 相关文件:
#     - config.env      : 统一配置文件，管理压测参数
#     - all_perf.py     : 实际执行压测的 Python 核心脚本
#     - unit_test_output: 压测结果保存的默认目录
# ==============================================================================
# =============================================================================
# Global Variable Declarations
# =============================================================================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU_DEVICE_LIST=""
START_TIME=0

# =============================================================================
# Section: Abstract Methods (Functions)
# =============================================================================

# Initialize environment and configurations
init_env() {
    source "${SCRIPT_DIR}/disable_local_proxy.sh"

    if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
        set -a
        source "${SCRIPT_DIR}/config.env"
        set +a
     else
        echo "[FATAL] config.env not found!" >&2
        exit 1
    fi

    export VLLM_PLUGINS=${VLLM_PLUGINS:-"fl"}
}

# Validate hardware and parse GPU list
validate_hardware() {
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[FATAL] CUDA_VISIBLE_DEVICES is unset." >&2
        exit 1
    fi

    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    GPU_DEVICE_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
}

# Lock GPU clocks
lock_gpu_clocks() {
    local lock_clock=${GPU_LOCK_CLOCK:-1830}
    echo "[INFO] Locking GPU clocks to ${lock_clock} MHz..."
    nvidia-smi -i "$GPU_DEVICE_LIST" -lgc "${lock_clock},${lock_clock}"
    nvidia-smi --query-gpu=index,name,clocks.gr --format=csv,noheader
}

# Reset GPU clocks
reset_gpu_clocks() {
    if [[ -n "$GPU_DEVICE_LIST" ]]; then
        echo "[INFO] Resetting GPU clocks..."
        nvidia-smi -i "$GPU_DEVICE_LIST" -rgc
        nvidia-smi --query-gpu=index,name,clocks.gr --format=csv,noheader
    fi
}

# Execute the benchmark
run_benchmark() {
    echo "[INFO] Starting benchmark..."
    local cpu_bind=${NUMA_CPUNODEBIND:-1}
    local mem_bind=${NUMA_MEMBIND:-1}

    # numactl --cpunodebind="${cpu_bind}" --membind="${mem_bind}" \
    # python3 "${SCRIPT_DIR}/all_perf.py" --input-len "${BENCH_INPUT_LEN:-4096}" --output-len "${BENCH_OUTPUT_LEN:-1024}" --concurrency "${BENCH_CONCURRENCY:-16}"
    
    # numactl --cpunodebind="${cpu_bind}" --membind="${mem_bind}" \
    # python3 "${SCRIPT_DIR}/all_perf.py" --input-len 32768 --output-len 1024 --concurrency 16
    
    numactl --cpunodebind="${cpu_bind}" --membind="${mem_bind}" \
    python3 "${SCRIPT_DIR}/all_perf.py" --input-len 16384 --output-len 1024 --concurrency 16
    
    numactl --cpunodebind="${cpu_bind}" --membind="${mem_bind}" \
    python3 "${SCRIPT_DIR}/all_perf.py" --input-len 4096 --output-len 1024 --concurrency 16
}

# Print summary
print_summary() {
    local end=$(date +%s)
    local duration=$((end - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "[INFO] Benchmark finished in ${minutes}m${seconds}s."
}

# =============================================================================
# Section: Main Flow Control
# =============================================================================
main() {
    init_env
    validate_hardware

    # Ensure GPU clocks are reset even if the script crashes or is Ctrl+C'd
    trap reset_gpu_clocks EXIT

    START_TIME=$(date +%s)

    lock_gpu_clocks
    run_benchmark

    # Explicitly reset clocks before finishing
    reset_gpu_clocks
    # Disable trap as we already reset successfully
    trap - EXIT

    print_summary
}

# Execution entry point
main "$@"
