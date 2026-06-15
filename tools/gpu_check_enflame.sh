#!/bin/bash

# Configuration parameters
mem_threshold=30000     # Maximum memory usage limit (MB)
sleep_interval=120          # Wait time (seconds), default is 2 minutes
max_wait=1200

# Sample output from efsmi -L
# DEV    SN                Slot    ID           Bus             CPU Affinity     UUID
# -------------------------------------------------------------------------------------------
# 0      A0A8640510030     N/A     1ea0:2a22    0000:23:00.0    0-31             TPUH61160105

# Count chip lines (lines with Bus info)
gpu_count=$(efsmi -L | grep -cP '\d+\:\d+\:')

if [ "$gpu_count" -eq 0 ]; then
    echo "No Enflame GPUs detected. Please ensure you have Enflame GPUs installed and properly configured."
    exit 1
fi

echo "Detected $npu_count Enflame NPU chip(s)."

waited_time=0
while true; do
    need_wait=false
    i=0

    printf " GPU  Total (MiB)  Used (MiB)  Free (MiB)\n"
    for ((i = 0 ; i < $gpu_count ; i++ )); do
	total_i=$(efsmi -i 0 -q -d MEMORY | awk '/Device Mem Info/,/BAR1/ { if (/Total Size/) {gsub(/[^0-9]/,"",$0); print $0}}')
	free_i=$(efsmi -i 0 -q -d MEMORY | awk '/Device Mem Info/,/BAR1/ { if (/Free Size/) {gsub(/[^0-9]/,"",$0); print $0}}')

        if [ -z "$free_i" ] || [ -z "$total_i" ]; then
            echo "Warning: Failed to parse memory infor for chip $i."
            continue
        fi

        used_i=$((total_i - free_i))

        printf "%4d%'13d%'12d%'12d\n" $i ${total_i} ${used_i} ${free_i}
        if [ $free_i -lt $mem_threshold ]; then
            need_wait=true
            break
        fi
    done

    if [ "$need_wait" = false ]; then
        echo "All GPUs have sufficient memory, proceeding with execution."
        break
    fi

    echo "GPU memory is insufficient, waiting for $sleep_interval seconds before retrying..."
    sleep $sleep_interval

    # Stop waiting if already waited too long
    waited_time=$((waited_time + sleep_interval))
    if [ $waited_time -gt $max_wait ]; then
        break
    fi
done
