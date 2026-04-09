#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_time=120          # Wait time (seconds)

# Check if hy-smi exists
if ! command -v hy-smi &> /dev/null; then
    echo "Error: hy-smi command not found. Please check if DTK is installed."
    exit 1
fi

# Get the number of DCUs
gpu_count=$(hy-smi 2>/dev/null | grep -c "Normal")

if [ "$gpu_count" -eq 0 ]; then
    echo "No Hygon DCU cards detected."
    exit 1
fi

echo "Detected $gpu_count Hygon DCU card(s)."

while true; do
    need_wait=false
    printf " DCU  Total (MiB)  Used (MiB)  Free (MiB)\n"

    for ((i=0; i<$gpu_count; i++)); do
        full_output=$(hy-smi 2>/dev/null)
        line=$(echo "$full_output" | awk -v idx="$i" '$1 == idx {print $0}')

        if [ -z "$line" ]; then
            echo "Warning: Failed to query DCU $i."
            need_wait=true
            break
        fi

        vram_percent=$(echo "$line" | awk '{print $6}' | tr -d '%')

        # Assume 32768 MiB (32GB) total VRAM for calculation
        total_i=32768
        used_i=$((total_i * vram_percent / 100))
        free_i=$((total_i - used_i))

        if [ "$vram_percent" == "0" ]; then
            free_i=$total_i
            used_i=0
        fi

        printf "%4d%'13d%'12d%'12d (VRAM: %d%%)\n" $i ${total_i} ${used_i} ${free_i} ${vram_percent}

        if [ $free_i -lt $mem_threshold ]; then
            need_wait=true
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All DCUs have sufficient memory."
        break
    fi

    echo "DCU memory is insufficient, waiting for $sleep_time seconds..."
    sleep $sleep_time
done
