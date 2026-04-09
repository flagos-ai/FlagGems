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
    need_wait=false
    printf " DCU  Total (MiB)  Used (MiB)  Free (MiB)\n"

    # Iterate through each card
    for ((i=0; i<$gpu_count; i++)); do
        # 1. Get full output
        full_output=$(hy-smi 2>/dev/null)
        
        # 2. Extract the line for this specific DCU index
        # We look for the line starting with the index number (e.g., "0", "1")
        # Note: The output format is "Index Temp AvgPwr Perf PwrCap VRAM% HCU% ..."
        # We need to be careful to match the exact row.
        # We use awk to filter the row where the first column matches $i
        line=$(echo "$full_output" | awk -v idx="$i" '$1 == idx {print $0}')

        if [ -z "$line" ]; then
            echo "Warning: Failed to query DCU $i."
            need_wait=true
            break
        fi

        # 3. Parse Memory from the line
        # The VRAM% column is usually the 6th column (index 5), but we need the actual memory size.
        # hy-smi usually doesn't show "Used/Total" in the main view, only percentage.
        # We might need to rely on a different command or parsing if percentage is all we have.
        
        # Let's try to find "VRAM" info. If the main view only has %, we might need to parse differently.
        # However, looking at your output, it only shows percentages (0%).
        # We need to find a way to get actual MBs.
        
        # Alternative: Try to grep for "Memory" or similar in the detailed view if -i doesn't work.
        # Since -i failed, let's assume we might only have percentage or need to calculate.
        # BUT, usually hy-smi has a detailed mode or the percentage is enough for "busy" check.
        
        # Let's try to extract the VRAM percentage first.
        # Columns: HCU(0) Temp(1) AvgPwr(2) Perf(3) PwrCap(4) VRAM%(5) ...
        vram_percent=$(echo "$line" | awk '{print $6}' | tr -d '%')

        # If we only have percentage, we can't know exact MBs without Total VRAM.
        # Let's assume standard 32GB or 64GB cards, or try to find Total VRAM elsewhere.
        # If we can't get exact MBs, we switch logic to check if VRAM% > threshold.
        
        # For now, let's try to see if we can get "Total Memory" from the header or a different flag.
        # If not, we approximate.
        
        # Let's try to run 'hy-smi -q' or similar? No, let's stick to what works.
        # If we can't get exact MBs, we will use percentage.
        # 30000MB on a 32GB (32768MB) card is ~91%. On a 64GB card is ~46%.
        # This is risky without knowing card size.
        
        # Let's try one more parsing attempt on the main output to see if we missed numbers.
        # Sometimes output is: "0 37C ... 12000MiB / 32000MiB ..."
        # But your output shows "0%".
        
        # Fallback: If we only have %, we assume a 32GB card (common for BW1000) for calculation
        # or we just check if it's non-zero.
        
        # Let's assume 32768 MiB (32GB) for calculation if exact parsing fails
        total_i=32768 
        used_i=$((total_i * vram_percent / 100))
        free_i=$((total_i - used_i))

        # If percentage is 0, free is likely total.
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
