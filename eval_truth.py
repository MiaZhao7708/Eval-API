import json, argparse, re
from tqdm import tqdm
from model import thread_parallel
from get_judge import get_judge
import os
import pandas as pd
from utils import utils
import numpy as np
import random

def rank_data(data, judge_method, i):
    judge = get_judge(judge_method)
    info_dict = {
        'prompt': data['prompt'],
        'response': data['response（有问题飘红）']
    }
    try:
        if f'S{i}' not in data.keys() or data[f'S{i}'] == '' or np.isnan(data[f'S{i}']):
            response_data = judge(info_dict=info_dict)
            if response_data:
                data[f'S{i}'] = response_data
                data[f'reason'] = info_dict['reason']
            else:
                return False
        return True
    except Exception as e:
        print(e)
        return False


def rank_file(judge_method, in_datas, out_file_name, thread_num, repeat_num):
    cache_prompts = {}
    if os.path.exists(out_file_name):
        out_datas = utils.read_data_file(out_file_name)
        for d in out_datas:
            cache_prompts[d['prompt']] = d
        in_datas = [cache_prompts.get(d['prompt'], d) for d in in_datas]
    
    print(f'len data = {len(in_datas)}')

    if not os.path.exists(os.path.dirname(out_file_name)):
        os.makedirs(os.path.dirname(out_file_name), exist_ok=True)

    for i in range(1, repeat_num + 1):
        call_status =thread_parallel(rank_data, in_datas, thread_num, extra_paras=(judge_method, i))
        failed = 0
        for status in call_status:
            if not status:
                failed += 1
        print(f'call eval wheel {i} done, failed rate = {failed / len(in_datas)}')
        random.shuffle(in_datas)
    for d in in_datas:
        scores = []
        for i in range(1, repeat_num + 1):
            if f'S{i}' in d:
                scores.append(d[f'S{i}'])
        d['score'] = 0 if len(scores) == 0 else sum(scores) / len(scores)

    out_df = pd.DataFrame.from_records(in_datas)
    out_df.to_excel(out_file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='call gpt4 to rank the generated responses')
    parser.add_argument('-i', '--input', type=str, help='the input json file')
    parser.add_argument('-o', '--output', default='output/outs.jsonl', type=str, help='the output jsonl file, default="output/outs.jsonl"')
    parser.add_argument('-l', '--jsonl', action='store_true', help='if the input file format is jsonl')
    parser.add_argument('-w', '--wheel', default=1, type=int, help='how many wheels to make the judge')
    parser.add_argument('-j', '--judge', default='bc_rm_model_judge', type=str, help='the judge method, default="bc_rm_model_judge"')
    parser.add_argument('-t', '--thread', default=40, type=int, help='the thread num when excuting the program, default="40"')
    args = parser.parse_args()
    
    in_datas = utils.read_data_file(args.input)
    if in_datas is not None and len(in_datas) > 0:
        rank_file(args.judge, in_datas, args.output, args.thread, args.wheel)