#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_time=120          # Wait time (seconds)

# Check if ppu-smi exists
if ! command -v ppu-smi &> /dev/null; then
    echo "Error: ppu-smi command not found."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(ppu-smi -L 2>/dev/null | grep -c "PPU-")

if [ "$gpu_count" -eq 0 ]; then
    echo "No T-Head PPU cards detected."
    exit 1
fi

echo "Detected $gpu_count T-Head PPU card(s)."

while true; do
    need_wait=false

    printf " PPU  Total (MiB)  Used (MiB)  Free (MiB)\n"

    # Iterate through each card
    for ((i=0; i<$gpu_count; i++)); do
        # 1. Get information for a single card
        # 2. Use grep -oP to extract the string in format "NumberMiB / NumberMiB"
        # 3. The regex matches: digits + MiB + whitespace + / + whitespace + digits + MiB
        mem_line=$(ppu-smi -i $i 2>/dev/null | grep -oP '\d+MiB\s*/\s*\d+MiB')

        if [ -z "$mem_line" ]; then
            echo "Warning: Failed to query PPU $i memory."
            need_wait=true
            break
        fi

        # Extract Used and Total
        # mem_line format example: 2MiB / 97920MiB
        used_i=$(echo "$mem_line" | grep -oP '^\d+')           # Extract digits at the beginning
        total_i=$(echo "$mem_line" | grep -oP '/\s*\K\d+')     # Extract digits after /

        if [ -z "$total_i" ] || [ -z "$used_i" ]; then
             echo "Warning: Parse error for PPU $i. Raw: '$mem_line'"
             need_wait=true
             break
        fi

        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}

        if [ $free_i -lt $mem_threshold ]; then
            need_wait=true
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All PPUs have sufficient memory."
        break
    fi

    echo "PPU memory is insufficient, waiting for $sleep_time seconds..."
    sleep $sleep_time
done
