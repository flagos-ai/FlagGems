#!/bin/bash

# Configuration
memory_usage_max=30000     # Maximum memory usage limit (MB)
sleep_time=120             # Wait time (seconds)

# 检查 ppu-smi 是否存在
if ! command -v ppu-smi &> /dev/null; then
    echo "Error: ppu-smi not found."
    exit 1
fi

# 获取 PPU 数量
# 统计包含 "PPU-" 的行（这是设备名称行的特征）
gpu_count=$(ppu-smi -L 2>/dev/null | grep -c "PPU-")

if [ "$gpu_count" -eq 0 ]; then
    echo "No PPUs detected."
    exit 1
fi

echo "Detected $gpu_count PPU(s)."
ppu-smi

while true; do
    need_wait=false

    for ((i=0; i<$gpu_count; i++)); do
        # 1. 获取单张 PPU 的信息
        # 2. 使用 awk 直接匹配包含 "MiB" 的行，不再依赖 "PPU" 关键词
        # 3. 提取显存信息 (格式: 2MiB / 97920MiB)
        
        # 获取包含显存信息的行
        mem_line=$(ppu-smi -i $i 2>/dev/null | awk '/MiB/ {print $0}')

        # 使用 sed 提取数字
        # 匹配模式：任意字符 + 数字 + MiB + / + 数字 + MiB
        memory_used=$(echo "$mem_line" | grep -oP '\d+(?=MiB \/)')
        memory_total=$(echo "$mem_line" | grep -oP '(?<=\/ )\d+(?=MiB)')

        # 检查是否获取成功
        if [ -z "$memory_used" ] || [ -z "$memory_total" ]; then
            echo "Warning: Failed to parse PPU $i memory. Raw line: '$mem_line'"
            need_wait=true
            break
        fi

        memory_remin=$((memory_total - memory_used))

        if [ "$memory_remin" -lt "$memory_usage_max" ]; then
            need_wait=true
            echo "PPU $i: Used ${memory_used}MB / Total ${memory_total}MB (Available: ${memory_remin}MB < ${memory_usage_max}MB)"
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All PPUs have sufficient available memory. Proceeding."
        break
    fi

    echo "PPU memory is insufficient, waiting for $sleep_time seconds..."
    sleep $sleep_time
done
