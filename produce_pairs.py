from model import thread_parallel
from utils.utils import read_data_file, write_jsonl, filter_input_prompts
import json
from get_judge import get_judge
import argparse
from glob import glob
import os
from copy import deepcopy
from rouge_score import rouge_scorer
from lingua import Language, LanguageDetectorBuilder
from rouge_chinese import Rouge
import re
from call_model_multiple_time import call_prompt


def judge_resp(data, judge):
    info_dict = {
        'prompt': data['prompt']
    }
    try:
        init_keys = [k for k in data.keys() if k.startswith('response_')]
        for k in init_keys:
            i = k.split('response_', 1)[-1]
            info_dict['response'] = data[k]
            response_data = judge(info_dict=info_dict)
            if response_data:
                data[f'M{i}'] = response_data
                data[f'reason_{i}'] = info_dict['reason']
            else:
                return None
        return data
    except Exception as e:
        print(e)
        return None


def filter_and_produce(data, judge, scorer, lan_checker, wheels):
    filter_prompt = '''请判断以下文本是不是一个用户向专家提的问题或要求。若是，且用户提的问题或要求需要清晰、目的且要求合理、无需专家进行额外澄清，则是一个合理指令。但若文本仅仅提供一个单纯的描述文本不含任何指令要求、或虽然是指令单用户却自己给出相关答案、或指令需求不合理专家无法完成，这就是一个不合理文本。请对先文本进行分析给出理由，再在最后使用【合理指令】或【不合理文本】来总结分类结果
    -----目标文本-------
{prompt}
    -----------------'''
    model = judge.get_model()
    prompt = data.get('prompt', None)
    
    if prompt is None:
        return None
    try:
        for _ in range(3):
            msg = filter_prompt.format(prompt=prompt)
            resp = model.call(msg)
            if resp:
                # here we don't return None for bad prompts because we may retry and should keep them to filter 
                if '【不合理文本】' in resp:
                    data['low_quality'] = True
                    return data
                elif '【合理指令】' in resp:
                    data['low_quality'] = False
                else:
                    continue
                if data['low_quality'] == False:
                    self_ins_model = deepcopy(model)
                    self_ins_model.set_temperature(1)
                    self_ins_model.set_top_k(50)
                    self_ins_model.set_top_p(0.99)
                    self_ins_model.set_stream(False)
                    out_d = call_prompt(data, self_ins_model, structure='prompt', rouge_scorer=scorer, lan_checker=lan_checker, top_k=wheels, extra_keys=['low_quality'])
                    #at least 2 resp
                    if 'response_1' in out_d:
                        out_d = judge_resp(out_d, judge)
                    return out_d
                return data
        return None
    except Exception as e:
        print(f'try to get response failed, the error is {e}')
    return None


def produce_responses(input_fos, out_file_name, judge_name, thread, wheels):
    input_datas = []
    for f in glob(input_fos + '/**/*.jsonl', recursive=True):
        data_part = read_data_file(f)
        input_datas.extend(data_part)

    tmp = []
    for d in input_datas:
        if '任务 9' in d['prompt'] or '任务9' in d['prompt'] or 'Task 9' in d['prompt']:
            task_list = re.split('(任务|Task|task)\s*\d+[:：]', d['prompt'])
            for t in task_list:
                t = t.strip()
                if len(t) > 8:
                    new_task = deepcopy(d)
                    new_task['prompt'] = t
                    tmp.append(new_task)
        else:
            tmp.append(d)
    input_datas = filter_input_prompts(tmp, out_file_name)
    judge = get_judge(judge_name)
    scorer = (rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True), Rouge())
    lan_checker = LanguageDetectorBuilder.from_all_spoken_languages().build()
    #for d in input_datas:
    #    out_d = filter_and_produce(d, judge, scorer, lan_checker, wheels)
    #    print(out_d)
    resps = thread_parallel(filter_and_produce, input_datas, threads=thread, extra_paras=(judge, scorer, lan_checker, wheels))
    write_jsonl(resps, out_file_name, mode='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self-instruct to get more similiar training pairs')
    parser.add_argument('-i', '--input', type=str, help='the input folder')
    parser.add_argument('-o', '--output', type=str, default='output/self_reward_with_resp', help='the output file to store the final res')
    parser.add_argument('-m', '--judge', type=str, default='bc_rm_model_truth', help="The judge used to score the resp")
    parser.add_argument('-t', '--thread', type=int, default=80, help='the thread num when excuting the program')
    parser.add_argument('-w', '--wheels', type=int, default=5, help='how many wheels want to ')
    args=parser.parse_args()

    produce_responses(args.input, args.output, args.judge, args.thread, args.wheels)