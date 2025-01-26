import json, argparse, re
from tqdm import tqdm
from model import thread_parallel
from get_judge import get_judge
import os
import pandas as pd
from utils import utils
import numpy as np
import random
import time, copy

def rank_data(data, judge_method, repeat_num):
    judge = get_judge(judge_method)
    info_dict = {
        'prompt': data['prompt'],
        'response': data['response（有问题飘红）']
    }
    avg_score = 0
    try:
        reasons = {}
        scores = 0
        for i in range(1, repeat_num + 1):
            if f'S{i}' not in data.keys() or data[f'S{i}'] == '' or np.isnan(data[f'S{i}']):
                response_data = judge(info_dict=info_dict)
                if response_data:
                    data[f'S{i}'] = response_data
                    scores += response_data
                    reasons[i] = info_dict['reason']
                else:
                    return False
                # sleep 0.1s to let other requests refresh cache
                time.sleep(0.1)
        avg_score = scores / repeat_num
        min_diff = 1e6
        data['score'] = avg_score
        for i, reason in reasons.items():
            diff = data[f'S{i}'] - avg_score
            if diff < min_diff:
                data['reason'] = reason
                min_diff = diff
    except Exception as e:
        print(e)
        return False
    if avg_score < 2.5:
        # use refelction to see if the model knows what it say
        judge_wrong= True
        gen_model = copy.deepcopy(judge.get_model())
        gen_model.set_temperature(0.5)
        history = [(judge.get_prompt().format_map(info_dict), data['reason'])]
        refelction_prompt = '那么你对于此指令正确的回答是：请以“正确的回答是：”为开头'
        try:
            # try to gen and judge repeat_num times, if there is a correct res, then we are sure the judge for original ans is correct
            for i in range(repeat_num):
                result = gen_model.call(refelction_prompt, history=history)
                if result is not None:
                    res_list = result.split('回答是：', 1)
                    if len(res_list) <= 1:
                        res_list = result.split('答案是：', 1)
                    if len(res_list) > 1:
                        res = res_list[1].strip()
                        info_dict = {
                            'prompt': data['prompt'],
                            'response': res
                        }
                        response_data = judge(info_dict=info_dict)
                        if response_data and response_data >= 4:
                            judge_wrong = False
                            data['correct_res'] = res
                            data['correct_score'] = response_data
                            break        
                    # sleep 0.1s to let other requests refresh cache
                    time.sleep(0.1)
        except Exception as e:
            print(e)
        # judge wrong, we hard code the score to 3
        if judge_wrong:
            data['score'] = 3
            data['correct_res'] = 'failed to pass reflection'
    return True


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

    call_status =thread_parallel(rank_data, in_datas, thread_num, extra_paras=(judge_method, repeat_num))
    failed = 0
    for status in call_status:
        if not status:
            failed += 1
    print(f'call eval done, failed rate = {failed / len(in_datas)}')

    out_df = pd.DataFrame.from_records(in_datas)
    out_df.to_excel(out_file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='call gpt4 to rank the generated responses')
    parser.add_argument('-i', '--input', type=str, help='the input json file')
    parser.add_argument('-o', '--output', default='output/outs.jsonl', type=str, help='the output jsonl file, default="output/outs.jsonl"')
    parser.add_argument('-l', '--jsonl', action='store_true', help='if the input file format is jsonl')
    parser.add_argument('-w', '--wheel', default=1, type=int, help='how many wheels to make the judge')
    parser.add_argument('-j', '--judge', default='bc_rm_model_judge', type=str, help='the judge method, default="bc_rm_model_judge"')
    parser.add_argument('-t', '--thread', default=100, type=int, help='the thread num when excuting the program, default="40"')
    args = parser.parse_args()
    
    in_datas = utils.read_data_file(args.input)
    if in_datas is not None and len(in_datas) > 0:
        rank_file(args.judge, in_datas, args.output, args.thread, args.wheel)