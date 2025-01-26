import os
import re
from tqdm import tqdm
import json
import pandas as pd
import glob
from get_judge import get_judge
from model import thread_parallel, get_model
import argparse
import numpy as np
from utils import utils
from eval_stability import cal_stability

def append_model_result(data, model, wheel_i):
    status = True
    answer = ''
    retry = 5
    while answer == '' and retry>= 0:
        try:
            answer = model.call(data['prompt'])
        except Exception:
            print(f'call model {model.name} error')
            answer = ''
            retry -= 1
    if retry <= 0:
        status = False
    data[f'ans_{model.name}_{wheel_i}'] = answer
    # return status for future get to ensure all future are processed and get success rate
    return status


def autojudge(data):
    try:
        judge = get_judge('gpt4_judge_no_answer')
        info_dict = {
            'prompt': data['prompt'],
            'data': data,
        }
        out_data = data.copy()
        # remove duplicate for lower api call cost
        ans_dict = {}
        for k, v in data.items():
            if k.startswith('ans_') and int(k.split('_')[-1]) < 6:
                out_k = 'truth_' + k[4:]
                if out_k not in out_data.keys():
                    if v not in ans_dict.keys():
                        ans_dict[v] = [k]
                    else:
                        ans_dict[v].append(k)
        for v, k_list in ans_dict.items():
            info_dict['response'] = v
            result = None
            retry_num = 5
            result = judge(info_dict=info_dict)
            if result is None:
                print(f'call gpt4 error')
                return None
            for k in k_list:
                k_suffix = k[4:]
                out_data['instruct_reason_' + k_suffix] = result['instruct_reason']
                out_data['answer_reason_' + k_suffix] = result['answer_reason']
                out_data['truth_' + k_suffix] = result['truth']
                out_data['rewrite_' + k_suffix] = result['rewrite']
    except Exception:
        return None
    return out_data

def eval_stability(model_names, input_data, output_name, gen_times, thread_num, detail_list):
    if not os.path.exists(os.path.dirname(output_name)):
        os.makedirs(os.path.dirname(output_name), exist_ok=True)

    for d in input_data:
        if ('prompt' not in d or '标准答案' not in d) and 'messages' in d and isinstance(d['messages'], list):
            messages = d['messages']
            d['prompt'] = messages[-2]['content']
            d['标准答案'] = messages[-1]['content']
    cache_file = os.path.splitext(output_name)[0] + '_model_out.jsonl'
    if os.path.exists(cache_file):
        cache = utils.read_data_file(cache)
        input_data = utils.filter_input_prompts(input_data, cache_file)
    else:
        cache = []
    print(f'len data = {len(input_data)}')
    
    model_list = [get_model(name) for name in model_names]
    if len(input_data) > 0:
        for wheel_i in range(1, gen_times + 1):
            for model in model_list:
                call_status = thread_parallel(append_model_result, input_data, threads=thread_num, name=model.name, extra_paras=(model, wheel_i))
                failed = 0
                for status in call_status:
                    if not status:
                        failed += 1
                print(f'wheel_{wheel_i}: call {model.name} done, failed rate = {failed / len(input_data)}')
                with open(cache_file + '.tmp', 'w', encoding="utf-8") as f_tmp:
                    json.dump(input_data, f_tmp, ensure_ascii=False, indent=4)
    cache.extend(input_data)
    utils.write_jsonl(cache, cache_file)
    # get gpt4 eval score
    gpt_cache_file = os.path.splitext(output_name)[0] + '_gpt4_eval.jsonl'
    if os.path.exists(gpt_cache_file):
        gpt_cache_prompts = {}
        with open(gpt_cache_file) as f_cache:
            for line in f_cache.readlines():
                gpt_cache_data = json.loads(line)
                if gpt_cache_data is not None and 'prompt' in gpt_cache_data.keys():
                    gpt_cache_prompts[gpt_cache_data['prompt']] = gpt_cache_data
        gpt_input_data = [gpt_cache_prompts.get(d['prompt'], d) for d in cache]
    else:
        gpt_input_data = cache
    print(f'gpt4 eval data = {len(gpt_input_data)}')
    out_eval_data = thread_parallel(autojudge, gpt_input_data, threads=thread_num)
    utils.write_jsonl(out_eval_data, gpt_cache_file)
    stability_eval_datas = pd.read_json(gpt_cache_file, lines=True)
    cal_stability(stability_eval_datas, model_list=model_names, group_name_list=detail_list, output_file_name=output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='call gpt4 to judge if the new response says the same this as prev response')
    parser.add_argument('-i', '--input', type=str, help='the input eval file')
    parser.add_argument('-o', '--output', type=str, default='output/judge_same/output.csv', help='the output file to store the judge result')
    parser.add_argument('-g', '--gen-times', type=int, default=10, help='how many generation answers per sample used to eval the stability')
    parser.add_argument('-d', '--details', type=str, default=None, help='If eval the stability for detailed dimension, split each dimension using comma for multi detailed dimensions')
    parser.add_argument('-m', '--models', type=str, default='bc_online_past,bc_online', help="The eval models, split each model using comma for multi model eval case")
    parser.add_argument('-t', '--thread', type=int, default=20, help='the thread num when excuting the program')
    args=parser.parse_args()

    model_names = args.models.split(',')
    detail_dimension = args.details
    detail_list = [] if detail_dimension is None else detail_dimension.strip().split(',')
    input_dict = utils.read_data_file(args.input)
    eval_stability(model_names, input_dict, args.output, args.gen_times, args.thread, detail_list)