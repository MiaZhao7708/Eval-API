import pandas as pd
import json
import os
from uuid import uuid4

def uuid(no_slash=False):
    u_id = str(uuid4())
    return u_id.replace('-', '') if no_slash else u_id

def write_jsonl(datas, output_file, mode='w'):
    failed = 0
    # here we don't use len(datas) to minus since it could be a future yields
    succeed = 0
    with open(output_file, mode, encoding='utf-8') as f_out:
        for r in datas:
            if r:
                f_out.write(json.dumps(r, ensure_ascii=False) + '\n')
                f_out.flush()
                succeed += 1
            else:
                failed += 1
    print(f'{succeed} datas writed to {output_file}, failed={failed}, failed_rate = {0 if failed == 0 else failed/(succeed + failed):.4f}')

def print_and_write(string, file, to_console=True):
    print(string, file=file)
    if to_console:
        print(string)

# get the list orderliness using the swap steps needed for sorting it using 'Bubble Sort' method. Eg: 
# [1, 4, 3, 2] is more orderly than [4, 2, 3, 1] because using 'Bubble Sort' method, the first one need 3 step, the last one need 5 step 
def sort_cost(input_list):
    length = len(input_list)
    if length <= 1:
        return 0
    # steps, steps(0) = 0,  steps(i + 1) = steps(i) + (i + 1) - target_pos(input_list[i + 1])
    sorted_list = [0] * length
    sorted_list[0] = input_list[0]
    steps = 0
    def find_pos_right_bound(target_list, num, left, right):
        mid = (left + right) // 2
        if mid == left:
            if target_list[right] <= num:
                return right + 1
            elif target_list[left] <= num:
                return right
            else:
                return left
        else:
            if target_list[mid] <= num:
                return find_pos_right_bound(target_list, num, mid, right)
            else:
                return find_pos_right_bound(target_list, num, left, mid)
    def insert_i_with_pos(target_list, num, current_list_len):
        pos = find_pos_right_bound(target_list, num, 0, current_list_len - 1)
        for j in range(current_list_len, pos, -1):
            target_list[j] = target_list[j - 1]
        target_list[pos] = num
        return pos
    for i in range(1, length):
        steps = steps + i - insert_i_with_pos(sorted_list, input_list[i], i)
    return steps

def read_data_file(input_file_name):
    file_suffix = input_file_name.strip().split('.')[-1]
    if file_suffix == 'json':
        input_data = pd.read_json(input_file_name)
    elif file_suffix == 'jsonl':
        input_data = pd.read_json(input_file_name, lines=True)
    elif file_suffix == 'csv':
        input_data = pd.read_csv(input_file_name)
    elif file_suffix == 'tsv':
        input_data = pd.read_csv(input_file_name, sep='\t')
    else:
        input_data = pd.read_excel(input_file_name)
    input_dict = input_data.to_dict('records')
    return input_dict


def get_req(d):
    assert isinstance(d, dict) or isinstance(d, list), 'The data should be dict format or list format!'
    if isinstance(d, list) or ('messages' in d and isinstance(d['messages'], list)):
        content = ''
        messages = d
        if not isinstance(d, list):
            messages = d['messages']
        for message in messages:
            if message['role'] == 'assistant':
                continue
            content += message['content']
        return content
    elif 'prompt' in d or 'Prompt' in d:
        syst = d.get('system', '')
        prompt = d.get('prompt', d.get('Prompt', None))
        return syst + prompt
    else:
        raise 'The data should contains "prompt" key or a standard chat message format'
        return None


def filter_input_prompts(input_datas, output_file):
    if os.path.exists(output_file):
        cache_prompts = set([])
        with open(output_file) as f_cache:
            for line in f_cache.readlines():
                d = json.loads(line)
                cache_prompts.add(get_req(d))
        out_datas = [d for d in input_datas if get_req(d) not in cache_prompts]
        return out_datas
    return input_datas