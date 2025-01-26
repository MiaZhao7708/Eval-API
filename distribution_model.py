import json, argparse, re
from tqdm import tqdm
from model import thread_parallel
from get_judge import get_judge
import os
import pandas as pd
from utils import utils
import numpy as np
import random

def label_data(data, judge):
    try:
        prompt = utils.get_req(data)
        if prompt is None or prompt == '':
            return None
        info_dict = {
            'prompt': prompt
        }
        resp = judge(info_dict=info_dict)
        if resp:
            data['cls_res'] = resp
            return data
    except Exception as e:
        print(f'call judge error, error={e}, data={data}')
    return None


def judge_file(in_datas, out_file_name, thread_num):

    cache_prompts = {}
    if os.path.exists(out_file_name):
        out_datas = utils.read_data_file(out_file_name)
        for d in out_datas:
            cache_prompts[utils.get_req(d)] = d
        in_datas = [cache_prompts.get(utils.get_req(d), d) for d in in_datas]
    
    print(f'len data = {len(in_datas)}')

    if not os.path.exists(os.path.dirname(out_file_name)):
        os.makedirs(os.path.dirname(out_file_name), exist_ok=True)

    if len(in_datas) > 0:
        utils.get_req(in_datas[0])
    judge = get_judge('distribution_model')
    call_results=thread_parallel(label_data, in_datas, thread_num, extra_paras=(judge, ))
    utils.write_jsonl(call_results, out_file_name, mode='a')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='call distribution model to classify responses')
    parser.add_argument('-i', '--input', type=str, help='the input json file')
    parser.add_argument('-o', '--output', type=str, help='the output jsonl file"')
    parser.add_argument('-t', '--thread', default=40, type=int, help='the thread num when excuting the program, default="40"')
    args = parser.parse_args()
    
    in_datas = utils.read_data_file(args.input)
    if in_datas is not None and len(in_datas) > 0:
        judge_file(in_datas, args.output, args.thread)