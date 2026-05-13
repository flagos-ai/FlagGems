#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_time=120          # Wait time (seconds), default is 2 minutes

gpu_count=$(mx-smi | awk '/Attached/ {print $4}' 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Failed to run mx-smi. Please check if mx-smi is installed and working correctly."
    exit 1
fi

if [ "$gpu_count" -eq 0 ]; then
    echo "No MetaX GPUs detected. Please ensure you have MetaX GPUs installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count MetaX GPU chip(s)."

while true; do
    memory_info=$(mx-smi | awk '/MiB[[:space:]]| A/ { print $9 }')
    need_wait=false
    readarray -t lines <<< "$memory_info"
    i=0

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for line in "${lines[@]}"; do
        used_i=$(echo "$line" | awk -F'/' '{print $1}')
        total_i=$(echo "$line" | awk -F'/' '{ print $2}')

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            continue
        fi

        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -lt $mem_threshold ]; then
            need_wait=true
            break
        fi
        i=$((i + 1))
    done

    if [ "$need_wait" = false ]; then
        echo "All GPUs have sufficient memory, proceeding with execution."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
