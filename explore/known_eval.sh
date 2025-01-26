#!/bin/bash

# CKPT
# ckpts=("llava-v1.5-7b-finetune-base-baseline" "llava-v1.5-7b-finetune-base-low" "llava-v1.5-7b-finetune-base-mid" "llava-v1.5-7b-finetune-base-high" "llava-v1.5-7b-finetune-base-full" "llava-v1.5-7b-finetune-ve-baseline" "llava-v1.5-7b-finetune-ve-low" "llava-v1.5-7b-finetune-ve-mid" "llava-v1.5-7b-finetune-ve-high" "llava-v1.5-7b-finetune-ve-full")
# ckpts=("llava-v1.5-7b-finetune-ve-full" "llava-v1.5-7b-finetune-base-full")
ckpts=("llava-v1.5-7b-finetune-ve-order-known" 'llava-v1.5-7b-finetune-ve-order')

function run_and_monitor() {
    local ckpt=$1
    local timeout=240  # 超时时间（秒）

    (python known_eval_bc.py $ckpt | tee /tmp/process_output.log) &

    local pid=$!  # PID
    local last_size=0
    local current_size
    local stuck_time=0

    while kill -0 $pid 2> /dev/null; do  # 确保进程仍在运行
        sleep 1  # 每秒检查一次文件大小
        current_size=$(stat -c %s /tmp/process_output.log)
        if [[ $current_size -eq $last_size ]]; then
            ((stuck_time++))
        else
            stuck_time=0
        fi

        if [[ $stuck_time -ge $timeout ]]; then
            echo "Progress has been stuck for over $timeout seconds, killing process..."
            kill -9 $pi
            wait $pid 2>/dev/null
            echo "Restarting the process..."
            run_and_monitor $ckpt  # 递归调用
            return
        fi

        last_size=$current_size
    done

    echo "Process completed for CKPT=$ckpt"
}

for ckpt in "${ckpts[@]}"; do
    echo "Running with CKPT=$ckpt"
    run_and_monitor $ckpt
done