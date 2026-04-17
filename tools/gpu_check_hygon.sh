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
# Count lines containing "Normal" or "C" (Temperature)
gpu_count=$(hy-smi 2>/dev/null | grep -c "Normal")

if [ "$gpu_count" -eq 0 ]; then
    echo "No Hygon DCU cards detected."
    exit 1
fi

echo "Detected $gpu_count Hygon DCU card(s)."

while true; do
    memory_usage=$(hy-smi --showmeminfo vram 2>/dev/null | awk 'BEGIN { FS=":" } /Used Memory/ { print $3 }')
    memory_total=$(hy-smi --showmeminfo vram 2>/dev/null | awk 'BEGIN { FS=":" } /Total Memory/ { print $3 }')

    # Check if hy-smi command was successful
    if [ $? -ne 0 ]; then
        echo "Failed to query GPU memory information. Please check if hy-smi is working correctly."
        exit 1
    fi

    # Convert query results to arrays
    IFS=$' ' read -d '' -r -a usage_array <<< "$memory_usage"
    IFS=$' ' read -d '' -r -a total_array <<< "$memory_total"

    need_wait=false

    printf " DCU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i=0; i<$gpu_count; i++)); do
        # trim 'M' from number
        total_i=${total_array[$i]}
        used_i=${usage_array[$i]}
        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
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
