#!/bin/bash

# Configuration
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_time=120             # Wait time (seconds), default is 2 minutes

export MUSA_INSTALL_PATH=/usr/local/musa
export PATH=$MUSA_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Get the number of GPUs
gpu_count=$(mthreads-gmi -L 2>/dev/null | grep -c "GPU ")

if [ "$gpu_count" -eq 0 ]; then
    echo "No Moore Threads GPUs detected. Please ensure you have GPUs installed and properly configured."
    exit 1
fi
echo "Detected $gpu_count Moore Threads GPU(s)."

while true; do
    need_wait=false

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    # Check the available memory for each GPU
    for ((i=0; i<$gpu_count; i++)); do
        # Query GPU memory information using mthreads-gmi
        memory_output=$(mthreads-gmi -q -d MEMORY -i $i 2>/dev/null)

        # Parse memory values from "FB Memory Usage" section
        # Format: "Total                                     :  81920MiB"
        total_i=$(echo "$memory_output" | grep -A 3 "FB Memory Usage" | grep "Total" | grep -oP '\d+' | head -1)
        used_i=$(echo "$memory_output" | grep -A 3 "FB Memory Usage" | grep "Used" | grep -oP '\d+' | head -1)

        # Check if we got valid memory values
        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to query GPU $i memory information."
            continue
        fi

        free_i=$((total_i - used_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -lt $mem_threshold]; then
            need_wait=true
            echo "GPU $i: Used ${used_i}MB / Total ${total}MB (Available: ${free_i}MB < ${mem_threshold}MB)"
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All Moore Threads GPUs have sufficient memory, proceeding with execution."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_time seconds before retrying..."
    sleep $sleep_time
done
