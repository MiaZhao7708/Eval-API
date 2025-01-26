import argparse
import json
import os
from model import get_model, thread_parallel
import copy
from utils import utils
from glob import glob

parser = argparse.ArgumentParser(description='call gpt4')
parser.add_argument('-i', '--input', type=str, nargs='*', help='the input eval files, split multiple file in space')
parser.add_argument('-o', '--output', type=str, default='output/online_log_correct.jsonl', help='the output file')
parser.add_argument('-m', '--model', type=str, default="gpt4-turbo", help='the thread_num to call')
parser.add_argument('-t', '--threads', type=int, default=50, help='the thread_num to call')
parser.add_argument('-s', '--structure', type=str, default='prompt', help='The output file data structure, support "prompt" and "conv"')
parser.add_argument('-e', '--extra-keys', type=str, default=None, help='If you want to add extra keys from original data to out data')
    
args=parser.parse_args()


def call_prompt(data, model, structure, extra_keys):
    if isinstance(data, list):
        in_data = {'messages': data}
    elif not ('messages' in data and isinstance(data['messages'], list)):
        in_data = {'messages': [{'role': 'user', 'content': data['prompt']}]}
    else:
        in_data = data
    try:
        res = model.process_data(in_data)
    except Exception as e:
        print(f'Call model error, error message is {e}')
        return None
    if res is not None:
        if structure == 'prompt':
            out_data = {}
            out_data['prompt'] = res['messages'][0]['content']
            out_data['response'] = res['messages'][1]['model']
        else:
            out_mes = []
            for mes in res['messages']:
                if 'model' in mes:
                    out_mes.append({'role': mes['role'], 'content': mes['model']})
                else:
                    out_mes.append({'role': mes['role'], 'content': mes['content']})
            out_data = {'messages': out_mes}
        if extra_keys is not None:
            for k in extra_keys:
                if k in data:
                    out_data[k] = data[k]
        return out_data
    return None


def get_model_result(model_name, input_list, output_file_name, structure, thread_num, extra_keys):
    model = get_model(model_name)
    if structure is None or structure not in {'prompt', 'conv'}:
        print('Unsupported output structure. Currently only support "prompt" and "conv"')
    datas = []
    for file_glob in input_list:
        for file_name in glob(file_glob, recursive=True):
            d_part = utils.read_data_file(file_name)
            datas.extend(d_part)
    #ensure data format is right
    if len(datas) > 0:
        utils.get_req(datas[0])
    gpt_input_data = utils.filter_input_prompts(datas, output_file_name)
    print(f'model eval data = {len(gpt_input_data)}')
    out_datas = thread_parallel(call_prompt, gpt_input_data, threads=thread_num, extra_paras=(model, structure, extra_keys))
    utils.write_jsonl(out_datas, output_file_name, mode='a')


if __name__ == '__main__':
    if isinstance(args.input, list):
        input_all_files = args.input
    else:
        input_all_files = [args.input]
    extra_keys = None if args.extra_keys is None else args.extra_keys.split(',')
    model = args.model
    get_model_result(model, input_all_files, args.output, args.structure, args.threads, extra_keys)