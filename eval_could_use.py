import json, argparse, re
from tqdm import tqdm
from model import thread_parallel
from get_judge import get_judge
from utils.utils import sort_cost, write_jsonl
import os
import copy
import numpy as np

def rank_data(data, i):
    judge = get_judge('bc_rm_model_judge')
    info_dict = {
        'prompt': data['Prompt']
    }
    try:
        k = 'R' + str(i)
        if f'M{i}' not in data.keys():
            info_dict['response'] = data[k]
            response_data = judge(info_dict=info_dict)
            if response_data:
                data[f'M{i}'] = response_data
                data[f'reason_{i}'] = info_dict['reason']
            else:
                return False
        return True
    except Exception as e:
        print(e)
        return False


def rank_file(judge_method, in_datas, out_file_name, thread_num):
    cache_prompts = {}
    if os.path.exists(out_file_name):
        with open(out_file_name) as f_cache:
            for line in f_cache.readlines():
                cache = json.loads(line)
                cache_prompts[cache['Prompt']] = cache
        in_datas = [cache_prompts.get(d['Prompt'], d) for d in in_datas]
    
    eval_data = []
    for data in in_datas:
        skip = False
        for i in range(1, 6):
            k = 'S' + str(i)
            if data[k] == 0:
                skip = True
                break
        if not skip:
            eval_data.append(data)
    print(f'len data = {len(eval_data)}')

    if not os.path.exists(os.path.dirname(out_file_name)):
        os.makedirs(os.path.dirname(out_file_name), exist_ok=True)

    for i in range(1, 6):
        call_status =thread_parallel(rank_data, eval_data, thread_num, extra_paras=(i,))
        failed = 0
        for status in call_status:
            if not status:
                failed += 1
        print(f'wheel_{i}: call eval done, failed rate = {failed / len(eval_data)}')
        with open(out_file_name + '.tmp', 'w', encoding="utf-8") as f_tmp:
            json.dump(eval_data, f_tmp, ensure_ascii=False, indent=4)
    write_jsonl(eval_data, out_file_name)
    out_datas = []
    with open(out_file_name) as f_read:
        for line in f_read.readlines():
            out_datas.append(json.loads(line))
    # get pass rate
    all_nums = {'all': 0}
    pos_nums = {'all': 0}
    for d in out_datas:
        all_nums['all'] += 1
        all_nums[d['细分能力项']] = all_nums.get(d['细分能力项'], 0) + 1
        if judge_best_good(d):
            pos_nums['all'] += 1
            pos_nums[d['细分能力项']] = pos_nums.get(d['细分能力项'], 0) + 1
    for k, v in all_nums.items():
        print(f'{k}: good rate = {pos_nums.get(k, 0)/v:.4f}, pos_num = {pos_nums.get(k, 0)}, all_num = {v}')
    # get avg sort cost
    costs = {'all': [0, 0]}
    for d in out_datas:
        target_sort_dict = {}
        target_sort_list = []
        for i in range(1, 6):
            target_sort_dict[f'M{i}'] = d[f'M{i}']
        for k, v in sorted(target_sort_dict.items(), key=lambda item: item[1]):
            target_sort_list.append(d['S' + k[1:]])
        c = sort_cost(target_sort_list)
        costs['all'][0] += c
        costs['all'][1] += 1
        if d['细分能力项'] not in costs.keys():
            costs[d['细分能力项']] = [0, 0]
        costs[d['细分能力项']][0] += c
        costs[d['细分能力项']][1] += 1
    
    for k, v in costs.items():
        print(f'{k}: avg sort cost = {v[0] / v[1]:.4f}')

    # get cos simility
    cos_sims = {'all': []}
    for d in out_datas:
        mul_res = 0
        vec_1 = 0
        vec_2 = 0
        for i in range(1, 6):
            mul_res += d[f'M{i}'] * d[f'S{i}']
            vec_1 += d[f'M{i}'] * d[f'M{i}']
            vec_2 += d[f'S{i}'] * d[f'S{i}']
        cos = mul_res / np.sqrt(vec_1) / np.sqrt(vec_2)
        cos_sims['all'].append(cos)
        if d['细分能力项'] not in cos_sims.keys():
            cos_sims[d['细分能力项']] = []
        cos_sims[d['细分能力项']].append(cos)

    for k, v in cos_sims.items():
        print(f'{k}: avg cos sim = {sum(v) / len(v):.4f}')    


def judge_best_good(data):
    best_i = 0
    best = -99
    for i in range(1, 6):
        if f'M{i}' in data and data[f'M{i}'] > best:
            best = data[f'M{i}']
            best_i = i
    return data[f'S{best_i}'] > 4

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='call gpt4 to rank the generated responses')
    parser.add_argument('-i', '--input', type=str, help='the input json file')
    parser.add_argument('-o', '--output', default='output/outs.jsonl', type=str, help='the output jsonl file, default="output/outs.jsonl"')
    parser.add_argument('-l', '--jsonl', action='store_true', help='if the input file format is jsonl')
    parser.add_argument('-j', '--judge', default='rm_ranker_with_cost', type=str, help='the judge method, default="rm_ranker_with_cost"')
    parser.add_argument('-t', '--thread', default=40, type=int, help='the thread num when excuting the program, default="40"')
    args = parser.parse_args()
    
    with open(args.input) as f_in:
        if args.jsonl:
            in_datas = []
            for line in f_in.readlines():
                in_datas.append(json.loads(line))
        else:
            in_datas = json.load(f_in)
    if in_datas is not None and len(in_datas) > 0:
        rank_file(args.judge, in_datas, args.output, args.thread)
    #prompts = '党中央对网络法治的要求在逐步向纵深迈进，依法治网的内涵和外延不断丰富和拓展。党中央对网络法治进行了一系列战略部署，明确了网络法治总体框架体系，确定了建立健全网络综合治理体系的目标，逐步将社会治理从现实社会向网络空间覆盖，网络法治的重要性和关键性不断凸显。 用口语化表达，缩写到80字内'
    #ans = '["网络法治不断深入发展，党中央明确了框架、目标，确保社会治理覆盖网络空间，提升网络法治的重要性和关键性。","随着互联网的普及，党中央对网络法治的要求也在不断提高。这包括明确网络法治的总体框架、建立网络综合治理体系，以及将社会治理扩展到网络空间。网络法治的重要性日益凸显，因为它关系到我们的日常生活和社会稳定。","随着网络的发展，党中央对网络法治的要求日益深化。明确总体框架、目标，强调其重要性和关键性，逐步实现社会治理从现实社会向网络空间覆盖。","随着网络发展，我国党中央对依法治网的要求越来越高。明确了网络法治体系，旨在构建网络综合治理。网络法治越来越被重视，逐渐覆盖现实社会和网络空间。","党中央对网络法治的重视不断增强，推动依法治网，其内涵丰富且外延广泛。在网络综合治理体系的引领下，我们正在逐步将社会治理扩展到网络空间。随着网络法治的关键性和重要性的不断凸显，它的影响越来越大。","互联网时代来临, 法律也紧跟科技发展同步进步适应新的生活模式与需求. 我们既要全面建立和完善\"网路治国\"的法律制度、机制及伦理道德规范,同时又要严密防范违法者有机可乘的网络犯罪行动或侵权行为; 这不但保护每个人的隐私权益和社会安全秩序,也为我們营造更公平正义的網絡環境提供了有力保障!"]'
    #print(eval_target)
    #print(rank(eval_target))
