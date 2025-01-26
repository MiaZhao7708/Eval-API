import os
import re
from tqdm import tqdm
import json
import pandas as pd
import glob
from model import thread_parallel
from get_judge import get_judge
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='call gpt4 to judge if the new response says the same this as prev response')
parser.add_argument('-i', '--input', type=str, help='the input json file')
parser.add_argument('-o', '--output', type=str, default='output/judge_same/output.csv', help='the output file to store the judge result')
parser.add_argument('-r', '--reference', type=str, help='the folder containing reference csv files')
parser.add_argument('-t', '--thread', type=int, default=20, help='the thread num when excuting the program')
args=parser.parse_args()

def autojudge(data, ref_df):
    judge = get_judge('judge_same')
    df_part = ref_df.loc[(ref_df.prompt == data['prompt']) & (ref_df.正确分 == 5)]
    if len(df_part) == 0:
        return data
    # randomlize choose one 5 star answer
    rand_index = np.random.randint(0, len(df_part))
    ref_data = df_part['response'].iloc[rand_index]
    info_dict = {
        'prompt': data['prompt'],
        'label': ref_data,
        'response': data['ans']
    }
    result = judge(info_dict=info_dict)
    if result is not None:
        data['score'] = result
        data['reason'] = info_dict['reason']
    return data


def convert_jsonl_to_csv(cache_file, ref_df, output_file):
    datas = []
    with open(cache_file) as f_in:
        for line in f_in.readlines():
            datas.append(json.loads(line))
    out_data = []
    for d in datas:
        score = d['score']
        d['正确分'] = d.pop('score')
        d['备注'] = d.pop('reason')
        d['response'] = d.pop('ans')
        df_part = ref_df.loc[(ref_df.prompt == d['prompt'])]
        if len(df_part) == 0:
            continue
        ref = df_part.iloc[0]
        d['id'] = ref.id
        d['sample'] = f'样本{len(df_part)}'
        d['source'] = ref.source
        d['语言1'] = ref.语言1
        d['语言2'] = ref.语言2
        d['指令约束'] = ref.指令约束
        d['难度'] = ref.难度
        d['能力'] = ref.能力
        d['属性'] = ref.属性
        d['领域'] = ref.领域
        d['标准答案'] = ref.标准答案
        d['回答形式'] = ''
        d['语言分'] = 5
        d['安全分'] = 1
        d['体感校验顺序'] = 6 - score
        d['放弃'] = ''
        d['8/2综合分'] = score * 0.8 + 1
        d['7/3综合分'] = score * 0.7 + 1.5
        d['6/4综合分'] = score * 0.6 + 2
        d['整体打分'] = f'{ref.标准答案}-{score}51-rank{6 - score}'
        d['标注人'] = ''
        d['check人'] = ''
        d['质检意见'] = ''
        d['质检备注'] = ''
        d['原标注员打分'] = ''
        d['标注时间'] = ''
        out_data.append(d)
    pd.DataFrame(out_data).to_csv(output_file, columns=['id', 'prompt', 'sample', 'source', 'response', '语言1', '语言2',
                                                        '指令约束', '难度', '能力', '属性', '领域', '标准答案', '回答形式',
                                                        '正确分', '语言分', '安全分', '体感校验顺序', '备注', '放弃', '8/2综合分',
                                                        '7/3综合分', '6/4综合分', '整体打分', '标注人', 'check人', '质检意见', '质检备注',
                                                        '原标注员打分', '标注时间'])




def judge_same(input_file, reference_folder, output_file, thread_num):
    reference_files = glob.glob(f'{reference_folder}/*.csv')
    df_list = []
    for file_name in reference_files:
        df_list.append(pd.read_csv(file_name))
    ref_df = pd.concat(df_list)

    with open(input_file) as f_in:
        input_data = json.load(f_in)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
   
    cache_file = os.path.splitext(output_file)[0] + '.jsonl'
    cache = set()
    if os.path.exists(cache_file):
        for line in open(cache_file):
            cache.add(json.loads(line)['prompt'])
        input_data = [v for k, v in input_data.items() if v['prompt'] not in cache]
    with open(cache_file, 'a') as f:
        results = thread_parallel(autojudge, dataset=input_data, threads=thread_num, name='auto_judge', extra_paras=(ref_df,))
        for r in results:
            if 'score' in r.keys():
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                f.flush()

    convert_jsonl_to_csv(cache_file, ref_df, output_file)
    

if __name__ == '__main__':
    datas = judge_same(args.input, args.reference, args.output, args.thread)
