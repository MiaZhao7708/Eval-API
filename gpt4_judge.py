import os
import re
import sys
import json
import logging
import threading
import pandas as pd
from get_judge import get_judge
from model import thread_parallel
from utils.utils import filter_input_prompts

# 配置模型名称
def autojudge(data):
    # for model in ('V5', 'V6', 'V7'):
    judge = get_judge('logic_math_two_stage')
    for model in ('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7'):
        info_dict = {
            'prompt': data['Prompt'],
            'label': data['标准答案'],
            'response': data[model]
        }
        result = judge(info_dict=info_dict)
        for key in ('可用性', '分数', '扣分原因'):
            data[f'{model}_{key}'] = result.get(key, '')
    return data


def run_autojudge(input_file='prompts/逻辑推理1205.xlsx', output_file='逻辑推理1205.xlsx'):
    dataset = [
        row.to_dict() for _, row in
        pd.read_excel(input_file, na_filter=False, dtype=str).iterrows()
    ]
   
    cache_file = os.path.splitext(input_file)[0] + '.jsonl'
    judge_count = len(dataset)
    dataset = filter_input_prompts(dataset, cache_file)
    judge_count -= len(dataset)

    failure_count = 0
    with open(cache_file, 'a') as f:
        results = thread_parallel(autojudge, dataset=dataset, threads=40, name='auto_judge')
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            judge_count += 1
            print(", ".join([
                f"当前进度:{judge_count}/{len(dataset)}",
                f"百分比:{judge_count / len(dataset) * 100:.2f}%",
                f"其中失败:{failure_count}"
            ]), flush=True)
            logging.info(f'Thread {threading.current_thread().name} is processed: '
                        f'{json.dumps(r, indent=2)}')      
            sys.stdout.flush()
    
    pd.DataFrame(dataset).to_excel(output_file, index=False)


def test_prompts():
    dataset = [
        row.to_dict() for _, row in
        pd.read_excel('prompts/逻辑推理1205.xlsx', na_filter=False, dtype=str).iterrows()
    ]
    print(autojudge(dataset[0], True))
    

if __name__ == '__main__':
    # test_prompts()
    run_autojudge('prompts/新的23条.xlsx', '新的23条.xlsx')
