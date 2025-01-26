import json, argparse, re
from tqdm import tqdm
from model import thread_parallel
from get_judge import get_judge
import os
import copy
from utils import utils

def rank_data(data, judge_method):
    judge = get_judge('rm_ranker_with_cost')
    info_dict = {
        'data': data,
        'instruction': data['input'],
        'answers': json.dumps(data['answers'], ensure_ascii=False, indent=2)
    }
    #ans_dict = {}
    #for i, v in enumerate(data['answers']):
    #    ans_dict[f'index {i}'] = v
    #info_dict = {
    #    'data': data,
    #    'instruction': data['input'],
    #    'answers': json.dumps(ans_dict, ensure_ascii=False, indent=2)
    #}
    try:
        response_data = judge(info_dict=info_dict)
        if response_data:
            out_data = response_data['result']
            out_data['cost'] = response_data['cost']
            return out_data
    except Exception as e:
        print(e)
        return None
    return None


def rank_file(judge_method, in_datas, out_file_name, thread_num):
    caches = []
    cache_prompts = set()
    if os.path.exists(out_file_name):
        with open(out_file_name) as f_cache:
            for line in f_cache.readlines():
                cache = json.loads(line)
                caches.append(cache)
                cache_prompts.add(cache['input'])
        in_datas = [d for d in in_datas if d['input'] not in cache_prompts]
    print(f'len data = {len(in_datas)}')
    all_res = thread_parallel(rank_data, in_datas, thread_num, extra_paras=(judge_method,))
    if not os.path.exists(os.path.dirname(out_file_name)):
        os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    utils.write_jsonl(all_res, out_file_name, 'a')


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
