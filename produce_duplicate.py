import argparse
import json
import os
from model import get_model, thread_parallel
from utils import utils
from glob import glob
import re, jieba

parser = argparse.ArgumentParser(description='call gpt4')
parser.add_argument('-i', '--input', type=str, nargs='*', help='the input eval files, split multiple file in space')
parser.add_argument('-o', '--output', type=str, default='output/online_log_correct.jsonl', help='the output file')
parser.add_argument('-m', '--model', type=str, default="bc_duplicate_gen", help='the thread_num to call')
parser.add_argument('-t', '--threads', type=int, default=50, help='the thread_num to call')
parser.add_argument('-k', '--keep', action='store_true', default=False, help='if keep the non dup data')
    
args=parser.parse_args()


def find_apear_time(a_str, sub):
    end = len(a_str)
    times = 0
    while True:
        end = a_str.rfind(sub, 0, end)
        if end == -1: return times
        times += 1


def check_resp_duplicate(response, n_gram = 10, repeat_num = 5, is_char_lan = False):
    # in english lans there are space between each words(which let list * 2) and duplicate words are more common, so here we multiple 2.5
    if is_char_lan:
        n_gram = int(n_gram * 2.5)
    resp_list = list(jieba.cut(response))
    resp_len = len(resp_list)
    if resp_len < n_gram:
        return False
    # we only check the last "5 * n-gram size" str instead of the total response to let process quicker,
    # since the response duplicate usually happens at the end of the response
    check_start = resp_len - 5 * n_gram - 1
    check_start = -1 if check_start < 0 else check_start
    for i in range(resp_len - 1 - n_gram, check_start, -1):
        gram_slice = ''.join(resp_list[i:i + n_gram])
        # ignore n-grams which have too many empty char
        if len(gram_slice.strip()) < n_gram / 2:
            continue
        times = find_apear_time(response, gram_slice)
        if times >= repeat_num:
            return True
    return False


def call_prompt(data, model, try_num=2):
    # we use this suffix to induce the model generate the response twice, with small temperature + weak sft model,
    # the model would easily go into duplicate generation (which is > 2)
    try:
        system = None
        prompt = None
        for message in data['messages']:
            if message['role'] == 'system' or message['role'] == 'user_system':
                system = message['content']
            elif message['role'] == 'user':
                prompt = message['content']
        # character based lan
        chn_lan_set = {'简中', '日语', '韩语', '其他中文', '繁中', '古汉语'}
        # word based lan
        eng_lan_set = {'英语', '西班牙语', '德语', '法语', '葡萄牙语'}
        # try twice
        for _ in range(try_num):
            res = model.call(prompt, system=system)
            if res is not None:
                lan = data['cls_res']['语言2'] if 'cls_res' in data else '简中'
                is_char_lan = lan in chn_lan_set
                repeat_num = 10 if lan in eng_lan_set else 8
                normal_resp_dup = check_resp_duplicate(res, repeat_num=repeat_num, is_char_lan=is_char_lan)
                data['model_response'] = res
                if normal_resp_dup:
                    data['model_resp_dup'] = normal_resp_dup
                    return data
        data['model_resp_dup'] = False
        return data
    except Exception as e:
        print(f'Call model error, error message is {e}')
        return None
    return data

def get_req(d):
    content = ''
    for message in d['messages']:
        if message['role'] == 'assistant':
            continue
        content_key = 'content' if message['role'] != 'assistant' else 'choose'
        content += message[content_key]
    return content

def get_model_result(model_name, input_list, output_file_name, thread_num, keep):
    model = get_model(model_name)
    datas = []
    for file_glob in input_list:
        for file_name in glob(file_glob, recursive=True):
            d_part = utils.read_data_file(file_name)
            datas.extend(d_part)
    if os.path.exists(output_file_name):
        gpt_cache_prompts = set([])
        with open(output_file_name) as f_cache:
            for line in f_cache.readlines():
                gpt_cache_data = json.loads(line)
                if gpt_cache_data is not None:
                    gpt_cache_prompts.add(get_req(gpt_cache_data))
        gpt_input_data = [d for d in datas if get_req(d) not in gpt_cache_prompts]
    else:
        gpt_input_data = datas
    print(f'model eval data = {len(gpt_input_data)}')
    out_datas = thread_parallel(call_prompt, gpt_input_data, threads=thread_num, extra_paras=(model,))
    with open(output_file_name, 'a') as f_out:
        success = 0
        for r in out_datas:
            if r and (r['model_resp_dup'] or keep):
                f_out.write(json.dumps(r, ensure_ascii=False) + '\n')
                f_out.flush()
                success += 1
                if success % 100 == 0 and success != 0:
                    print(f'{success} dup datas processed')


if __name__ == '__main__':
    if isinstance(args.input, list):
        input_all_files = args.input
    else:
        input_all_files = [args.input]
    model = args.model
    get_model_result(model, input_all_files, args.output, args.threads, args.keep)