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
#     启动 vLLM 服务端。脚本会自动读取同级目录下的 config.env 配置文件，
#     支持动态开启 Profiler (Trace) 或 Eager 模式，并包含服务健康检查。
#
# 前置条件:
#     1. 确保当前目录下存在 config.env 文件，且其中配置了 MODEL_PATH。
#     2. 建议在启动前通过环境变量显式指定 GPU，如不指定，将默认使用全部 GPU。
#
# 使用方法:
#     # 方式一：默认启动（脚本会自动在后台拉起服务并轮询健康状态）
#     bash start.sh
#
#     # 方式二：指定特定的 GPU 启动（推荐在多卡机器上使用）
#     export CUDA_VISIBLE_DEVICES=0,1
#     bash start.sh
#
#     # 方式三：临时覆盖配置（不修改 config.env 文件，仅本次生效）
#     VLLM_QWEN3_PORT=8000 ENABLE_PROFILER=false bash start.sh
#
# 相关文件:
#     - config.env: 统一配置文件
#     - pid.txt   : 服务启动后记录的进程 PID 文件
#     - vllm.log  : vLLM 运行日志输出文件
# ==============================================================================
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(pwd -P)"

# External config variables (assigned in init_env)
LOG_FILE=""
PID_FILE=""
PROFILER_CONFIG_JSON=""
VLLM_ARGS=()

# =============================================================================
# Section: Abstract Methods (Functions)
# =============================================================================

# Initialize environment and configurations
init_validate_env() {
    source "${SCRIPT_DIR}/disable_local_proxy.sh"
    source ~/env.sh

    if [[ -f "${SCRIPT_DIR}/config.env" ]]; then
        source "${SCRIPT_DIR}/config.env"
    else
        echo "[FATAL] config.env not found!" >&2
        exit 1
    fi
    
    if [[ -z "${MODEL_PATH:-}" ]]; then
        echo "[FATAL] MODEL_PATH is missing. Please specify MODEL_PATH in config.env." >&2
        exit 1
    fi

    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "[WARNING] CUDA_VISIBLE_DEVICES is not set. Defaulting to all available GPUs." >&2
    fi
    
    LOG_FILE=${VLLM_LOG_FILE:-"vllm.log"}
    PID_FILE=${PID_FILE:-"pid.txt"}
    
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    export VLLM_QWEN3_PORT=${VLLM_QWEN3_PORT:-7000}
    export VLLM_PLUGINS=${VLLM_PLUGINS:-"fl"}
    export USE_FLAGGEMS=${USE_FLAGGEMS:-1}
}

# Print environment and version info
print_versions() {
    echo "[INFO] vLLM $(python3 -m pip show vllm 2>/dev/null | grep Version || true)"
    echo "[INFO] vllm-plugin-fl $(python3 -m pip show vllm-plugin-FL 2>/dev/null | grep Version || true)"
    echo "[INFO] Torch $(python3 -m pip show torch 2>/dev/null | grep Version || true)"
    echo "[INFO] FlagGems $(python3 -m pip show flag_gems 2>/dev/null | grep Version || true)"
    echo "[INFO] FlagTree $(python3 -m pip show flagtree 2>/dev/null | grep Version || true)"
}

# Clean up old services and processes
cleanup_old_service() {
    if [[ -f "$PID_FILE" ]]; then
        local old_pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            local process_dir=$(pwdx "$old_pid" 2>/dev/null); process_dir=${process_dir#*: }
            if [[ "$process_dir" == "$CURRENT_DIR" ]]; then
                echo "[WARNING] Service already running: pid = $old_pid."
                bash "${SCRIPT_DIR}/stop.sh"
            else
                echo "[WARNING] pid = $old_pid belongs to '${process_dir:-unknown}', skip."
            fi
        fi
    fi
    bash "${SCRIPT_DIR}/clear_fuser_process.sh"
}

# Build Profiler JSON string
build_profiler_config() {
    local warmup_sec=""
    local active_sec=""
    if [[ -n "${PROFILER_WARMUP_ITERS:-}" ]]; then
        warmup_sec=$(printf ',\n    "warmup_iterations": %s' "${PROFILER_WARMUP_ITERS}")
    fi
    
    if [[ -n "${PROFILER_ACTIVE_ITERS:-}" ]]; then
        active_sec=$(printf ',\n    "active_iterations": %s' "${PROFILER_ACTIVE_ITERS}")
    fi
    PROFILER_CONFIG_JSON=$(cat <<EOF
{
    "profiler": "torch",
    "torch_profiler_dir": "${PROFILER_DIR:-${SCRIPT_DIR}}",
    "torch_profiler_record_shapes": ${PROFILER_RECORD_SHAPES:-false},
    "torch_profiler_with_stack": ${PROFILER_RECORD_STACK:-false},
    "torch_profiler_with_memory": ${PROFILER_RECORD_MEMORY:-false},
    "torch_profiler_with_flops": ${PROFILER_RECORD_FLOPS:-false},
    "torch_profiler_dump_cuda_time_total": ${PROFILER_DUMP_CUDA_TIME_TOTAL:-false},
    "ignore_frontend": ${IGNORE_FRONTEND:-true}${warmup_sec}${active_sec}
}
EOF
)
}
# Build vllm serve argument array
build_vllm_args() {
    VLLM_ARGS=(
        "${MODEL_PATH}"
        "--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE:-10}"
        "--port" "${VLLM_QWEN3_PORT:-7000}"
        "--served-model-name" "${SERVED_MODEL_NAME:-qwen}"
        "--max-model-len" "${MAX_MODEL_LEN:-65536}"
        "--trust-remote-code"
        "--gpu-memory-utilization" "${GPU_MEM_UTIL:-0.8}"
        "--dtype" "${DTYPE:-bfloat16}"
    )

    if [[ "${ENABLE_EAGER:-false}" == "true" ]]; then
        echo "[INFO] Eager mode is ENABLED."
        echo "[INFO] Graph mode is DISABLE."
        VLLM_ARGS+=("--enforce-eager")
    else
        echo "[INFO] Eager mode is DISABLE."
        echo "[INFO] Graph mode is ENABLED"
    fi

    if [[ "${ENABLE_PROFILER:-false}" == "true" ]]; then
        echo "[INFO] Profiler is ENABLED."
        build_profiler_config
        VLLM_ARGS+=("--profiler-config" "${PROFILER_CONFIG_JSON}")
    else
        echo "[INFO] Profiler is DISABLED."
    fi
}

# Start service process
start_service() {
    echo "[INFO] Starting vllm serve..."
    echo "[INFO] Command: numactl --cpunodebind=${NUMA_CPUNODEBIND:-1} --membind=${NUMA_MEMBIND:-1} \\"
    echo "[INFO] nohup vllm serve ${VLLM_ARGS[*]} > ${LOG_FILE} 2>&1 &"
    
    numactl --cpunodebind=${NUMA_CPUNODEBIND:-1} --membind=${NUMA_MEMBIND:-1} \
    nohup vllm serve "${VLLM_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
    
    echo "$!" > "${PID_FILE}"
}

# Poll for health check
wait_for_startup() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[FATAL] $PID_FILE not found!"; exit 1
    fi

    local pid=$(head -n 1 "$PID_FILE" | tr -d '[:space:]')
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "[FATAL] PID = $pid is invalid!"; exit 1
    fi

    local max_retry=${MAX_RETRY:-180}
    local retry_interval=${RETRY_INTERVAL_SEC:-10}
    local start_time=$(date +%s)

    echo "[INFO] Waiting for service to be ready..."
    for ((i=1; i<=max_retry; i++)); do
        sleep "${retry_interval}"

        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[FATAL] Process $pid died unexpectedly. Check log: ${LOG_FILE}"
            exit 1
        fi

        if bash "${SCRIPT_DIR}/ping.sh" 2>/dev/null; then
            echo ""
            echo "[INFO] Service startup successfully."
            return 0
        fi

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        echo "[INFO] Service startup elapsed time: ${minutes}m${seconds}s"
    done

    echo "[FATAL] Detection failed! Reached max retry: ${max_retry}"
    echo "[FATAL] Please check log: ${LOG_FILE}"
    exit 1
}

# =============================================================================
# Section: Main Flow Control
# =============================================================================
main() {
    init_validate_env
    print_versions
    cleanup_old_service
    build_vllm_args
    start_service
    wait_for_startup
}

# Execution entry point
main "$@"