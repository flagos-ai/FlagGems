#!/bin/bash

# Configuration
memory_usage_max=30000     # Maximum memory usage limit (MB) - 设置显存剩余阈值
sleep_time=120             # Wait time (seconds), default is 2 minutes

# 平头哥显卡（如阿里云 gn7e 等）通常兼容 NVIDIA CUDA 生态，使用 nvidia-smi 进行管理
# 确保 nvidia-smi 在 PATH 中，通常位于 /usr/bin/nvidia-smi

# 检查 nvidia-smi 是否存在
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
# 使用 nvidia-smi -L 列出 GPU 并统计行数
gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c "GPU")

if [ "$gpu_count" -eq 0 ]; then
    echo "No GPUs detected. Please ensure you are on a valid T-Head GPU instance."
    exit 1
fi

echo "Detected $gpu_count GPU(s) (T-Head/NVIDIA Compatible)."
nvidia-smi

while true; do
    need_wait=false

    # Check the available memory for each GPU
    for ((i=0; i<$gpu_count; i++)); do
        # Query GPU memory information using nvidia-smi
        # We query memory.total and memory.used in MiB
        memory_info=$(nvidia-smi -i $i --query=memory=total,used --format=csv,noheader,nounits 2>/dev/null)
        
        # Parse the output (Format: "Total MiB, Used MiB")
        memory_total=$(echo "$memory_info" | awk -F ',' '{print $1}' | tr -d ' ')
        memory_used=$(echo "$memory_info" | awk -F ',' '{print $2}' | tr -d ' ')

        # Check if we got valid memory values
        if [ -z "$memory_used" ] || [ -z "$memory_total" ]; then
            echo "Warning: Failed to query GPU $i memory information."
            continue
        fi

        memory_remin=$((memory_total - memory_used))

        if [ $memory_remin -lt $memory_usage_max ]; then
            need_wait=true
            echo "GPU $i: Used ${memory_used}MB / Total ${memory_total}MB (Available: ${memory_remin}MB < ${memory_usage_max}MB)"
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All GPUs have sufficient available memory. Proceeding with execution."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
