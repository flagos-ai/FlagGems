#!/bin/bash

# Configuration parameters
memory_usage_max=30000     # Maximum memory usage limit (MB)
sleep_time=120             # Wait time (seconds), default is 2 minutes

# Get the number of NPU chips from npu-smi info output
# Chip lines look like: "| 0     0                   | 0000:9D:00.0  | 0           0    / 0          2894 / 65536         |"
# Count lines that contain HBM usage pattern "xxxx / xxxxx" at the end (the HBM-Usage column)
npu_smi_output=$(npu-smi info 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Failed to run npu-smi. Please check if npu-smi is installed and working correctly."
    exit 1
fi

# Count chip lines (lines with Chip/Phy-ID and HBM usage info)
chip_count=$(echo "$npu_smi_output" | grep -cP '\d+\s*/\s*\d+\s*\|\s*$')
# Each NPU card has 2 chips, but we check per-chip
npu_count=$(echo "$npu_smi_output" | grep -c "Ascend")

if [ "$npu_count" -eq 0 ]; then
    echo "No Ascend NPUs detected. Please ensure you have Ascend NPUs installed and properly configured."
    exit 1
fi

echo "Detected $npu_count Ascend NPU chip(s)."

echo "$npu_smi_output"

while true; do
    npu_smi_output=$(npu-smi info 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "Failed to query NPU information. Please check if npu-smi is working correctly."
        exit 1
    fi

    # Parse HBM-Usage from chip lines
    # Chip line example: "| 0     0                   | 0000:9D:00.0  | 0           0    / 0          2894 / 65536         |"
    # Each chip line has 3 "xxx / yyy" patterns: Hugepages-Usage, Memory-Usage, HBM-Usage
    # We extract the last "xxx / yyy" from each chip line (HBM-Usage)
    mapfile -t hbm_lines < <(echo "$npu_smi_output" | grep "0000:" | while IFS= read -r line; do
        echo "$line" | grep -oP '\d+\s*/\s*\d+' | tail -1
    done)

    need_wait=false
    chip_idx=0

    for line in "${hbm_lines[@]}"; do
        memory_used=$(echo "$line" | awk -F'/' '{gsub(/[[:space:]]/, "", $1); print $1}')
        memory_total=$(echo "$line" | awk -F'/' '{gsub(/[[:space:]]/, "", $2); print $2}')

        if [ -z "$memory_used" ] || [ -z "$memory_total" ]; then
            echo "Warning: Failed to parse chip $chip_idx memory information."
            chip_idx=$((chip_idx + 1))
            continue
        fi

        memory_remin=$((memory_total - memory_used))

        if [ $memory_remin -lt $memory_usage_max ]; then
            need_wait=true
            echo "Chip $chip_idx: Used ${memory_used}MB / Total ${memory_total}MB (Available: ${memory_remin}MB < ${memory_usage_max}MB)"
            break
        fi
        chip_idx=$((chip_idx + 1))
    done

    if [ "$need_wait" = false ]; then
        echo "All Ascend NPUs have sufficient available memory. Proceeding with execution."
        break
    fi

    echo "NPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
