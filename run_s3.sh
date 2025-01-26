#!/bin/bash
ckpts=("hr-llava-v1.5-8b-landmark-finetune-best-q64" "hr-llava-v1.5-8b-landmark-finetune-best-q144")
splits=("test")
part="391"

for ckpt in "${ckpts[@]}"; do
    for split in "${splits[@]}"; do
        # 提取 lora 后面的第一个单词作为 part
        # part=$(echo $ckpt | cut -d'-' -f6)
         
        echo "Running for checkpoint: $ckpt, split: $split, part: $part"
        
        # 调用 python 脚本
        python gpt4_landmark_judge.py $ckpt $split $part
    done
done
