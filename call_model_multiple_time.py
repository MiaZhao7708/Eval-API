import argparse
import json
import os
from model import get_model, thread_parallel
from utils import utils
from glob import glob
from rouge_score import rouge_scorer # type: ignore
from lingua import Language, LanguageDetectorBuilder # type: ignore
from rouge_chinese import Rouge # type: ignore
import jieba # type: ignore

def call_prompt(data, model, extra_keys, top_k, rouge_scorer, lan_checker):
    # currently only process turn 1
    syst, prompt = None, None
    if isinstance(data, list) or ('messages' in data and isinstance(data['messages'], list)):
        mes_list = data
        if not isinstance(data, list):
            mes_list = data['messages']
        for mes in mes_list:
            if mes['role'] in {'system', 'user_system'}:
                syst = mes['content']
            elif mes['role'] == 'user':
                prompt = mes['content']
    else:
        syst, prompt = data.get('system', None), data['prompt']
    if prompt is None:
        return None
    try:
        res = model.get_top_responses(prompt, top_k, system=syst)
    except Exception as e:
        print(f'Call model error, error message is {e}')
        return None
    out_d = {
        'system': '' if syst is None else syst,
        'prompt': prompt,
    }
    if 'category' in data:
        out_d['cls'] = data['category']
    elif 'cls_res' in data:
        out_d['cls'] = data['cls_res']
    elif 'cls' in data:
        out_d['cls'] = data['cls']
    else:
        out_d['lan'] = str(lan_checker.detect_language_of(prompt))
    for k in extra_keys:
        if k in data.keys():
            out_d[k] = data[k]
    if res is not None and len(res) > 0:
        filter_can = []
        for r in res:
            filter_can.append((r, lan_checker.detect_language_of(r) in {Language.CHINESE, Language.JAPANESE, Language.KOREAN}))
        if len(filter_can) > 0:
            filter_can = remove_duplicate(rouge_scorer, list(set(filter_can)))
        for i, r in enumerate(filter_can):
            out_d[f'response_{i}'] = r
        return out_d
    return None

def remove_duplicate(scorer, candidates_pair_list):
    out_list = [candidates_pair_list[0][0]]
    for can_pair in candidates_pair_list[1:]:
        if pass_rouge_list(scorer, can_pair, out_list):
            out_list.append(can_pair[0])
    return out_list

def pass_rouge_list(scorer, prompt_cjk_pair, target_list):
    for s_v in target_list:
        prompt, is_cjk = prompt_cjk_pair
        if is_cjk:
            h = ' '.join(jieba.cut(prompt))
            v = ' '.join(jieba.cut(s_v))
            s = scorer[1].get_scores(h, v)[0]['rouge-l']['f']
        else:
            rouge_L = scorer[0].score(prompt, s_v)
            s = rouge_L['rougeL'].fmeasure
        if s > 0.9:
            return False
    return True

def get_model_result(model_name, input_list, output_file_name, thread_num, extra_keys, wheels):
    model = get_model(model_name, temperature=1, timeout=400)
    model.set_top_k(50)
    model.set_top_p(0.99)
    datas = []
    for file_glob in input_list:
        for file_name in glob(file_glob, recursive=True):
            d_part = utils.read_data_file(file_name)
            datas.extend(d_part)
    #ensure data format is right
    if len(datas) > 0:
        utils.get_req(datas[0])
    gpt_input_data = utils.filter_input_prompts(datas, output_file_name)
    scorer = (rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True), Rouge())
    lan_checker = LanguageDetectorBuilder.from_all_spoken_languages().build()
    print(f'model eval data = {len(gpt_input_data)}')
    out_datas = thread_parallel(call_prompt, gpt_input_data, threads=thread_num, extra_paras=(model, extra_keys, wheels, scorer, lan_checker))
    utils.write_jsonl(out_datas, output_file_name, mode='a')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='call gpt4')
    parser.add_argument('-i', '--input', type=str, nargs='*', help='the input eval files, split multiple file in space')
    parser.add_argument('-o', '--output', type=str, default='output/online_log_correct.jsonl', help='the output file')
    parser.add_argument('-m', '--model', type=str, default="bc_self_ins", help='the thread_num to call')
    parser.add_argument('-t', '--threads', type=int, default=50, help='the thread_num to call')
    parser.add_argument('-e', '--extra-keys', type=str, default=None, help='If you want to add extra keys from original data to out data')
    parser.add_argument('-w', '--wheels', type=int, default=5, help='how many wheels want to ')
    
    args=parser.parse_args()

    if isinstance(args.input, list):
        input_all_files = args.input
    else:
        input_all_files = [args.input]
    extra_keys = None if args.extra_keys is None else args.extra_keys.split(',')
    model = args.model
    get_model_result(model, input_all_files, args.output args.threads, extra_keys, args.wheels)
    