import json
from utils.utils import read_data_file
import argparse
import random
import re
from model import get_model, thread_parallel
from rouge_chinese import Rouge
import os
import copy
import jieba

def get_model_res(input_data, model, limit, scorer):
    prompt_instruct_raw = 'Please generate a instruct similar to the following instruct. The core meaning stay unchanged but the ask way is changed as different as possiable. You should directly generate the instruct with no any other explanation\n--- orignal instruct ---\n'
    prompt = prompt_instruct_raw + input_data['question'] + '\n--- end of orignal instruct ---'
    gen_ans = 0
    out_list = []
    r = ' '.join(jieba.cut(input_data['question']))
    while gen_ans < limit:
        try:
            response = model.call_details(prompt)
            if response is not None and response.get('status', '') == 'finish' and 'text' in response:
                res = response['text']
                h = ' '.join(jieba.cut(res))
                maxrougeL = scorer.get_scores(h, r)[0]['rouge-l']['f']
                for o_res in out_list:
                    o_r = ' '.join(jieba.cut(o_res['prompt']))
                    rougeL = scorer.get_scores(h, o_r)[0]['rouge-l']['f']
                    if maxrougeL < rougeL:
                        maxrougeL = rougeL
                if maxrougeL < 0.75 or (len(res) > 600 and maxrougeL < 0.95):
                    out_data = {'raw': input_data['question'], 'prompt': res, 'reference': input_data['reference']}
                    out_list.append(out_data)
                    gen_ans += 1
        except Exception as e:
            print(e)
            return None
    return out_list



def gen_re_phrased_datas(seed_model, input_datas, output_name, limit, thread_num):
    caches = set([])
    model = get_model(seed_model)
    cache_file = os.path.splitext(output_name)[0] + '_raw_instruct.jsonl'
    if os.path.exists(cache_file):
        with open(cache_file) as f_cache:
            for line in f_cache.readlines():
                cache_d = json.loads(line)
                caches.add(cache_d['old_response'])
    input_datas = [d for d in input_datas if d['question'] not in caches]
    print(len(input_datas))
    scorer = Rouge()

    with open(cache_file, 'a') as f_out:
        for res in thread_parallel(get_model_res, input_datas, threads=thread_num, name=model.name, extra_paras=(model, limit, scorer)):
            if res is not None:
                for r in res:
                    out_data = {'old_response': r['raw'], 'prompt': r['prompt'], 'reference': r['reference']}
                    f_out.write(json.dumps(out_data, ensure_ascii=False) + '\n')
                    f_out.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self-instruct to get more similiar qa pairs')
    parser.add_argument('-i', '--input', type=str, help='the input eval file')
    parser.add_argument('-o', '--output', type=str, default='output/align_bench/out.jsonl', help='the output file to store the final res')
    parser.add_argument('-l', '--limit', type=int, default=2, help='how many re-phrased data you want')
    parser.add_argument('-m', '--model', type=str, default='gpt4-turbo', help="The seed gen model")
    parser.add_argument('-t', '--thread', type=int, default=50, help='the thread num when excuting the program')
    args=parser.parse_args()

    input_datas = read_data_file(args.input)

    gen_re_phrased_datas(args.model, input_datas, args.output, args.limit, args.thread)