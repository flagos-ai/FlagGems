#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_interval=120      # Wait time (seconds), default is 2 minutes
max_wait=1200           # Maximum wait time (seconds), default is 20 minutes

# Get the number of Sunrise chips from pt_smi output
# Chip lines look like:
# |   0  SR-SUN-S2-X1-PCIE   N/A  | 00000000:8D:00.0     N/A |                    0 |
# | N/A   34C    P7    89W / 400W |            0B / 62847MiB |       0%         N/A |
# Count lines that contain HBM usage pattern "xxxx / xxxxx" at the end (the HBM-Usage column)
gpu_count=$(pt_smi | grep -cP '\d+B / \d+MiB')
echo $gpu_count

if [ $? -ne 0 ]; then
    echo "Failed to run pt_smi. Please check if pt_smi is installed and working correctly."
    exit 1
fi

if [ "$gpu_count" -eq 0 ]; then
    echo "No Sunrise GPUs detected. Please ensure you have Sunrise GPUs installed and properly configured."
    exit 1
fi

echo "Detected $gpu_count Sunrise GPU chip(s)."

waited_time=0
while true; do
    smi_output=$(pt_smi | grep -oP '\d+B / \d+MiB')

    need_wait=false
    i=0

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    pt_smi | grep -oP '\d+B / \d+MiB' | while read -r line; do
        used_i=$(echo "$line" | grep -oP '^\d+')
	total_i=$(echo "$line" | grep -oP '\d+(?=MiB)')

        if [ -z "$used_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to parse memory infor for chip $i."
            i=$((i + 1))
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
    sleep $sleep_interval
    waited_time=$(( waited_time + sleep_interval ))
    if [ $waited_time -gt $max_wait ]; then
        break
    fi
done
