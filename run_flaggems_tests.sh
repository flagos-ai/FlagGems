#!/bin/bash

# ---------------- config ----------------
FLAGGEMS_PATH="/PATH/OF/YOUR/FLAGGEMS"
OP_LIST="/PATH/OF/YOUR/OPS_LIST_FILE/" #Operator list file, one operator per line

# Specify GPU list
GPUS=(0 1 2 3 4 5 6 7)
TIMESTAMP=$(date +"%Y%m%d_%H%M")
RESULTS_DIR="${FLAGGEMS_PATH}/results_${TIMESTAMP}"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"

mkdir -p "$RESULTS_DIR"
chmod u+rwx "$RESULTS_DIR"

echo "operator,accurancy_passed,accurancy_failed,accurancy_skipped,accurancy_total,avg_speedup,float16_speedup,float32_speedup,bfloat16_speedup,int16_speedup,int32_speedup,bool_speedup,cfloat_speedup" >> "$SUMMARY_FILE"

GPU_COUNT=${#GPUS[@]}
echo "result dir: $RESULTS_DIR"
echo

run_op_test() {
    local op="$1"
    local gpu_id="$2"


    op=$(echo "$op" | tr -d '\r' | xargs)
    if [ -z "$op" ]; then
        return
    fi

    local op_dir="${RESULTS_DIR}/${op}"
    mkdir -p "$op_dir"
    chmod u+rwx "$op_dir"

    echo "[GPU $gpu_id] start op: $op"

    # ---------------- accurancy ----------------
    local acc_log="${op_dir}/accuracy.log"
    # Note: adjustments are required based on different GPUs.
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    bash -c "cd \"$FLAGGEMS_PATH/tests\" && pytest -m \"$op\" --ref cpu" \
    >"$acc_log" 2>&1

    local acc_exit=$?
    [ $acc_exit -eq 0 ] && acc_res="PASS" || acc_res="FAIL"

    # Extract pytest results
    local summary_line
    summary_line=$(grep -E "={3,8} .* in .*s.*={3,8}" "$acc_log" | tail -n 1)
    local passed=0 failed=0 skipped=0 total=0
    if [[ -n "$summary_line" ]]; then
        [[ "$summary_line" =~ ([0-9]+)[[:space:]]+passed ]] && passed=${BASH_REMATCH[1]}
        [[ "$summary_line" =~ ([0-9]+)[[:space:]]+failed ]] && failed=${BASH_REMATCH[1]}
        [[ "$summary_line" =~ ([0-9]+)[[:space:]]+skipped ]] && skipped=${BASH_REMATCH[1]}
        total=$((passed + failed + skipped))
    fi
    # ---------------- benchmark test ----------------
    local perf_log="${op_dir}/perf.log"

    find "${FLAGGEMS_PATH}/benchmark" -maxdepth 1 \
    -type f -name "result-m_${op}--level_core--record_log.log" \
    -exec rm -f {} +
    # Note: adjustments are required based on different GPUs.
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    bash -c "cd \"$FLAGGEMS_PATH/benchmark\" && pytest -m \"$op\" --level core --record log" \
    >"$perf_log" 2>&1

    local perf_result_file
    perf_result_file=$(find "${FLAGGEMS_PATH}/benchmark" -maxdepth 1 -type f -name "result-m_${op}--level_core--record_log.log" | head -n 1)
    if [ -n "$perf_result_file" ]; then
        mv "$perf_result_file" "$op_dir/"
        perf_result_file="${op_dir}/$(basename "$perf_result_file")"
    else
        perf_result_file="N/A"
    fi

    # ---------------- parse result ----------------
    local avg_speedup="0"
    local float16_speedup="0"
    local float32_speedup="0"
    local bfloat16_speedup="0"
    local int16_speedup="0"
    local int32_speedup="0"
    local bool_speedup="0"
    local cfloat_speedup="0"

    if [ -f "${FLAGGEMS_PATH}/benchmark/summary_for_plot.py" ] && [ -f "$perf_result_file" ]; then
        parsed_summary_log="${op_dir}/parsed_summary.log"
        rm -f "$parsed_summary_log"
        python "${FLAGGEMS_PATH}/benchmark/summary_for_plot.py" "$perf_result_file" > "$parsed_summary_log" 2>&1

        if [ -s "$parsed_summary_log" ]; then
            read float16_speedup float32_speedup bfloat16_speedup int16_speedup int32_speedup bool_speedup cfloat_speedup avg_speedup < <(
                awk -v target="$op" '
                BEGIN{
                    IGNORECASE = 1
                }
                function trim(s) {
                    gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s)
                    # remove surrounding quotes if any
                    if (s ~ /^".*"$/) { sub(/^"/, "", s); sub(/"$/, "", s) }
                    return s
                }
                NR==1 {
                    for(i=1;i<=NF;i++){
                        h = tolower(trim($i))
                        gsub(/[ \t]+/, "", h)
                        headers[i]=h
                    }
                    next
                }
                {
                    opname = tolower(trim($1))
                    tt = tolower(target)
                    matched = 0
                    if (opname == tt) matched = 1
                    else if (opname == tt "_") matched = 1
                    else if (index(opname, tt) == 1) matched = 1
                    if (!matched) next


                    f16 = f32 = bf16 = i16 = i32 = bl = cf = avg = "0"
                    for(i=1;i<=NF;i++){
                        h = headers[i]
                        val = trim($(i))
                        if (h == "float16_speedup") f16 = val
                        else if (h == "float32_speedup") f32 = val
                        else if (h == "bfloat16_speedup") bf16 = val
                        else if (h == "int16_speedup") i16 = val
                        else if (h == "int32_speedup") i32 = val
                        else if (h == "bool_speedup") bl = val
                        else if (h == "cfloat_speedup") cf = val
                        else if (h == "avg_speedup" || h == "average_speedup" || h == "avg") avg = val
                    }

                    printf "%s %s %s %s %s %s %s %s\n", f16, f32, bf16, i16, i32, bl, cf, avg
                    exit
                }
                END {

                    printf "0 0 0 0 0 0 0 0\n"
                }' "$parsed_summary_log"
            )


            float16_speedup=$(echo "$float16_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            float32_speedup=$(echo "$float32_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            bfloat16_speedup=$(echo "$bfloat16_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            int16_speedup=$(echo "$int16_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            int32_speedup=$(echo "$int32_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            bool_speedup=$(echo "$bool_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            cfloat_speedup=$(echo "$cfloat_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')
            avg_speedup=$(echo "$avg_speedup" | sed 's/^"//;s/"$//;s/^[[:space:]]*//;s/[[:space:]]*$//')

            is_number_regex="^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$"
            if [ -z "$avg_speedup" ] || [ "$avg_speedup" = "0" ]; then
                sum=0; cnt=0
                for v in "$float16_speedup" "$float32_speedup" "$bfloat16_speedup" "$int16_speedup" "$int32_speedup" "$bool_speedup" "$cfloat_speedup"; do
                    if [[ $v =~ $is_number_regex ]]; then
                        gt=$(awk -v x="$v" 'BEGIN{ print (x+0>0) ? 1 : 0 }')
                        if [ "$gt" -eq 1 ]; then
                            sum=$(awk -v a="$sum" -v b="$v" 'BEGIN{ printf "%.12f", a + b }')
                            cnt=$((cnt+1))
                        fi
                    fi
                done
                if [ "$cnt" -gt 0 ]; then
                    avg_speedup=$(awk -v s="$sum" -v c="$cnt" 'BEGIN{ printf "%.6f", s / c }')
                else
                    avg_speedup="0"
                fi
            fi
        fi
    fi


    echo "✅ [GPU $gpu_id] $op finished (acc:$acc_res | p:$passed f:$failed s:$skipped t:$total | speedup:$avg_speedup)"

    echo "$op,$passed,$failed,$skipped,$total,$avg_speedup,$float16_speedup,$float32_speedup,$bfloat16_speedup,$int16_speedup,$int32_speedup,$bool_speedup,$cfloat_speedup" >> "$SUMMARY_FILE"




}

# ---------------- main ----------------
OPS=()
while IFS= read -r line || [ -n "$line" ]; do
    line=$(echo "$line" | tr -d '\r' | xargs)
    if [ -n "$line" ] && [[ ! "$line" =~ ^# ]]; then
        OPS+=("$line")
    fi
done < "$OP_LIST"

if [ ${#OPS[@]} -eq 0 ]; then
    echo "no ops in file"
    exit 1
fi

declare -A GPU_TASKS
for ((i=0; i<${#OPS[@]}; i++)); do
    gpu_index=$((i % GPU_COUNT))
    gpu_id=${GPUS[$gpu_index]}
    GPU_TASKS[$gpu_id]="${GPU_TASKS[$gpu_id]} ${OPS[$i]}"
done

# ----------------do test ----------------
for gpu_id in "${GPUS[@]}"; do
    {
        for op in ${GPU_TASKS[$gpu_id]}; do
            run_op_test "$op" "$gpu_id"
        done
    } &
done

wait

echo "-------------------------------------------"
echo "result dir: ${RESULTS_DIR}"
echo "summary file: ${SUMMARY_FILE}"
echo "-------------------------------------------"
