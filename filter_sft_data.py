import argparse, json
from utils import utils
import os
import pandas as pd
from glob import glob
from get_judge import get_judge
from model import thread_parallel
import random, copy, time


def remove_nan(d):
    out_d = {}
    for k, v in d.items():
        if isinstance(v, str) or isinstance(v, list) or isinstance(v, dict) or not pd.isnull(v):
            out_d[k] = v
    return out_d


def process_data(data, judge, repeat_num):
    mes = data['messages']
    system = None
    prompt = None
    response = None
    for m in mes:
        if m['role'] == 'assistant':
            if response is None:
                response = m['content']
            else:    
                print(f'bad data!!! data detail is: {data}')
                return data
        elif m['role'] == 'user':
            if prompt is None:
                prompt = m['content']
            else:
                print(f'bad data!!! data detail is: {data}')
                return data
        # currently only use first system
        elif (m['role'] == 'system' or m['role'] == 'user_system') and system is None:
            system = m['content']
    if prompt is None and response is None:
        print(f'bad data!!! data detail is: {data}')
        return data
    if system is not None:
        prompt = '系统设定：\n' + system + '\n\n正式指令：\n' + prompt 
    info_dict = {
        'prompt': prompt,
        'response': response
    }

    out_data = copy.deepcopy(data)
    avg_score = 0
    for i in range(1, repeat_num + 1):
        try:
            response_data = judge(info_dict=info_dict)
            if response_data:
                out_data[f'S{i}'] = response_data
                data[f'S{i}'] = response_data
                avg_score += response_data
                data[f'R{i}'] = info_dict['reason']
                # sleep 0.1s to let other requests refresh cache
                time.sleep(0.1)
            else:
                return data
        except Exception as e:
            print(e)
            return data
    out_data['score'] = avg_score / repeat_num
    min_diff = 1e6
    for i in range(1, repeat_num + 1):
        diff = abs(out_data[f'S{i}'] - out_data['score']) 
        if diff < min_diff:
            min_diff = diff
            out_data['reason'] = data[f'R{i}']
    return out_data

def get_req(data):
    reqs = ''
    for mes in data['messages']:
        if mes['role'] != 'assistant':
            reqs += mes['content']
    return reqs

def process_file(judge, score, data_file, out_file, thread, repeat_num):
    out_split = out_file.rsplit('/', 1)
    filtered_fo = out_split[0] + '/filtered/'
    # the dir of out_file would also be made by this action
    if not os.path.exists(filtered_fo):
        os.makedirs(filtered_fo, exist_ok=True)
    filtered_file = filtered_fo + out_split[1]
    cache = set([])

    in_datas = utils.read_data_file(data_file)
    if os.path.exists(out_file):
        out_datas = utils.read_data_file(out_file)
        for d in out_datas:
            cache.add(get_req(d))
    if os.path.exists(filtered_file):
        out_datas = utils.read_data_file(filtered_file)
        for d in out_datas:
            cache.add(get_req(d))

    target_in_datas = []
    with open(out_file, 'a') as f_out, open(filtered_file, 'a') as f_filter:
        for d in in_datas:
            if get_req(d) in cache:
                continue
            d['original_file'] = out_file
            turn = 0
            for mes in d['messages']:
                if mes['role'] in {'user', 'assistant'}:
                    turn += 1
            # don't treat turns larger than 2
            if turn != 2:
                out_d = remove_nan(d)
                f_out.write(json.dumps(out_d, ensure_ascii=False) + '\n')
                f_out.flush()
            else:
                target_in_datas.append(d)
        if len(target_in_datas) > 0:
            print(f'start process {data_file}, process length={len(target_in_datas)}')
            out_datas = thread_parallel(process_data, target_in_datas, thread, extra_paras=(judge, repeat_num))

            failed = 0
            for d in out_datas:
                if d.get('score', 0) != 0:
                    out_d = remove_nan(d)
                    if d['score'] > score:
                        f_out.write(json.dumps(out_d, ensure_ascii=False) + '\n')
                        f_out.flush()
                    else:
                        f_filter.write(json.dumps(out_d, ensure_ascii=False) + '\n')
                        f_filter.flush()
                else:
                    failed += 1
            print(f'call eval done for {data_file}, failed={failed}, failed rate = {failed / len(target_in_datas):.5f}')


def filter_dataset(config, input_folder, output_folder, thread, repeat_num):
    judge = get_judge(config['judge'])
    for dataset in config['dataset']:
        score = dataset['score']
        d_fos = dataset['dir']
        for d_fo in d_fos:
            data_fo = os.path.join(input_folder, d_fo)
            if os.path.isfile(data_fo):
                print(d_fo)
                out_f = output_folder + '/' + d_fo.split(input_folder, 1)[-1]
                process_file(judge, score, data_fo, output_folder, thread, repeat_num)
            elif os.path.isdir(data_fo):
                for f in glob(data_fo + '/**/*.jsonl', recursive=True):
                    out_f = output_folder + '/' + f.split(input_folder, 1)[-1]
                    process_file(judge, score, f, out_f, thread, repeat_num)
            else:
                for f in glob(data_fo + '*.jsonl', recursive=True):
                    out_f = output_folder + '/' + f.split(input_folder, 1)[-1]
                    process_file(judge, score, f, out_f, thread, repeat_num)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='filter potention bad sft datas use judge model')
    parser.add_argument('-i', '--input', type=str, help='the input dir')
    parser.add_argument('-o', '--output', type=str, default='output/sft_data_filter/', help='the output folder to store the filter result')
    parser.add_argument('-c', '--config', type=str, default='config/filter_score.json', help='the filter config')
    parser.add_argument('-t', '--thread', type=int, default=20, help='the thread num when excuting the program')
    parser.add_argument('-w', '--wheel', type=int, default=3, help='how many wheels do you need')
    args=parser.parse_args()

    config = {}
    with open(args.config) as f_c:
        config = json.load(f_c)
    if config:
        filter_dataset(config, args.input, args.output, args.thread, args.wheel)