#!/usr/bin/env bash
set -euo pipefail

# 统一时间戳，方便看是同一轮实验
TS="$(date +%Y%m%d_%H%M%S)"
LOG_WINOGRAD="resnet34_winograd_${TS}.log"
LOG_TOEPLITZ="resnet34_toeplitz_${TS}.log"

# 自定义 time 输出格式（可选）
# %R = real, %U = user CPU, %S = sys CPU
TIMEFORMAT=$'\n[elapsed] real=%R sec, user=%U sec, sys=%S sec\n'

run_mode() {
    local mode="$1"
    local log="$2"

    echo "==================================================" >>"$log"
    echo "Mode: $mode" >>"$log"
    echo "Start time: $(date)" >>"$log"
    echo "--------------------------------------------------" >>"$log"

    # 用 bash 内建的 time 统计运行时间
    { time taskset -c 0 python3 resnet34.py --mode "$mode"; } >>"$log" 2>&1

    echo "--------------------------------------------------" >>"$log"
    echo "End time:   $(date)" >>"$log"
    echo >>"$log"
}

run_mode "winograd" "$LOG_WINOGRAD"
run_mode "toeplitz" "$LOG_TOEPLITZ"

