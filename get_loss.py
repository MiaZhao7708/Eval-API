import argparse
import json
from utils.utils import read_data_file, write_jsonl
from session import session
from model import thread_parallel
import random
import os

def process_data(data):
    url = 'http://resp-loss-calc.gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com/generate'
    try:
        resp = session.post(url=url, json={'query': data}, timeout=60)
        res = resp.json()
        if res['code'] == 200:
            return res['data']['response_item']
    except Exception as e:
        print(f'call model error, error is {e}')
    return None

def get_req(data):
    reqs = ''
    for mes in data['messages']:
        if mes['role'] != 'assistant':
            reqs += mes['content']
        else:
            if 'choose' in mes and mes['choose'] is not None:
                reqs += mes['choose']
            if 'reject' in mes and mes['reject'] is not None:
                reqs += mes['reject']
    return reqs


def get_loss(in_datas, out_file_name, thread):
    cache_prompts = set([])
    if os.path.exists(out_file_name):
        out_datas = read_data_file(out_file_name)
        for d in out_datas:
            cache_prompts.add(get_req(d))
        in_datas = [d for d in in_datas if get_req(d) not in cache_prompts]
    random.shuffle(in_datas)
    out_datas = thread_parallel(process_data, in_datas, thread)
    write_jsonl(out_datas, out_file_name, 'a')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='call gpt4')
    parser.add_argument('-i', '--input', type=str, nargs='*', help='the input eval files, split multiple file in space')
    parser.add_argument('-o', '--output', type=str, default='out_merged.jsonl', help='the output file')
    parser.add_argument('-t', '--threads', type=int, default=30, help='the thread_num to call')
    
    args=parser.parse_args()

    if isinstance(args.input, list):
        input_all_files = args.input
    else:
        input_all_files = [args.input]
    input_data = []
    for in_file in input_all_files:
        input_data.extend(read_data_file(in_file))
    get_loss(input_data, args.output, args.threads)