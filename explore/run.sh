#!/bin/bash

# 定义CKPT
# ckpts=("llava-v1.5-7b-finetune-base-baseline" "llava-v1.5-7b-finetune-base-low" "llava-v1.5-7b-finetune-base-mid" "llava-v1.5-7b-finetune-base-high" "llava-v1.5-7b-finetune-base-full" "llava-v1.5-7b-finetune-ve-baseline" "llava-v1.5-7b-finetune-ve-low" "llava-v1.5-7b-finetune-ve-mid" "llava-v1.5-7b-finetune-ve-high" "llava-v1.5-7b-finetune-ve-full")
ckpts=("llava-v1.5-7b-finetune-ve-order-known" 'llava-v1.5-7b-finetune-ve-order' "llava-v1.5-7b-finetune-ve-order-reverse-unknown" "llava-v1.5-7b-finetune-ve-order-reverse")
# 循环访问CKPT
for ckpt in "${ckpts[@]}"; do
    python known_eval_bc.py $ckpt
done