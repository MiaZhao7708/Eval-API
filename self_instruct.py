import json
from utils.utils import read_data_file
import argparse
import random
import re
from model import get_model, thread_parallel
import os
from rouge_score import rouge_scorer
from lingua import Language, LanguageDetectorBuilder
from rouge_chinese import Rouge
import copy
import jieba

lan_checker = LanguageDetectorBuilder.from_all_spoken_languages().build()

def get_model_res(prompt_pair, model):
    k, prompt = prompt_pair
    pat = '\n*\s*Task\s*\d+:\s*\n*'
    try:
        response = model.call_details(prompt)
        if response is not None and response.get('status', '') == 'finish' and 'text' in response:
            res = response['text']
            return k, re.split(pat, res)
    except Exception as e:
        print(e)
        return None, None
    return None, None


def str_is_cjk(input_str):
    lan = lan_checker.detect_language_of(input_str)
    return lan in {Language.CHINESE, Language.JAPANESE, Language.KOREAN}


def list_rouge_filter(scorer, target_list):
    if len(target_list) < 2:
        return target_list
    out_list = [target_list[0]]
    cmp_list = [target_list[0][1]]
    for i in range(1, len(target_list)):
        if len(target_list[i][1]) > 10 and 'image' not in target_list[i][1] and 'picture' not in target_list[i][1] \
        and '图片' not in target_list[i][1] and '视频' not in target_list[i][1] and pass_rouge_list(scorer, (target_list[i][1], target_list[i][2]), cmp_list):
            out_list.append(target_list[i])
            cmp_list.append(target_list[i][1])
    return out_list


def pass_rouge_list(scorer, prompt_cjk_pair, target_list, threshold=0.7):
    prompt, is_cjk = prompt_cjk_pair
    for v in target_list:
        if is_cjk:
            h = ' '.join(jieba.cut(prompt))
            v_split = ' '.join(jieba.cut(v))
            s = scorer[1].get_scores(h, v_split)[0]['rouge-l']['f']
        else:
            rouge_L = scorer[0].score(prompt, v)
            s = rouge_L['rougeL'].fmeasure
        if s > threshold:
            return False
    return True


def pass_rouge_dict(scorer, prompt_cjk_pair, target_dict):
    for v_list in target_dict.values():
        pass_rouge = pass_rouge_list(scorer, prompt_cjk_pair, v_list)
        if not pass_rouge:
            return False
    return True


def check_rougeL(prompt_pair, scorer, instruction_type_datas_in, instruction_type_datas_out):
    try:
        k, prompt, is_cjk = prompt_pair
        if pass_rouge_dict(scorer, (prompt, is_cjk), instruction_type_datas_in) and pass_rouge_dict(scorer, (prompt, is_cjk), instruction_type_datas_out):
            return k, prompt
    except Exception as e:
        print(e)
    return None, None


def gen_self_instruct_datas(seed_model, input_datas, output_name, limit, thread_num):
    prompt_self_instruct_raw = 'Please generate few more new tasks similiar to the format constraint like the following tasks but different task goal. Each new task should contain the similiar comma constraint in the provided task. You should generate start with \'Task 9: \'\n---- Examples ----'
    instruction_type_datas_in = {}
    instruction_type_datas_out = {}
    out_datas_len = 0
    model = get_model(seed_model)
    if os.path.exists(output_name):
        with open(output_name) as f_cache:
            for line in f_cache.readlines():
                cache_d = json.loads(line)
                if cache_d['instruct_group'] not in instruction_type_datas_out:
                    instruction_type_datas_out[cache_d['instruct_group']] = []
                instruction_type_datas_out[cache_d['instruct_group']].append(cache_d['prompt'])
                out_datas_len += 1
    for d in input_datas:
        ins_id_list = d['instruct_group_list']
        for i in ins_id_list:
            #if 'no_comma' not in i:
            #    continue
            if i not in instruction_type_datas_in:
                instruction_type_datas_in[i] = []
            instruction_type_datas_in[i].append(d["prompt"])
    for k, v in instruction_type_datas_in.items():
        instruction_type_datas_in[k] = list(set(v))
    #(eng_scorer, cjk_scorer)
    scorer = (rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True), Rouge())

    empty_times = 0
    while True:
        #try 30 times for empty_times
        if out_datas_len > limit or empty_times > 30:
            break
        task_pool = []
        for k, v in instruction_type_datas_in.items():
            datas_for_prompt_raw = random.sample(v, 8)
            data_not_used = random.sample(datas_for_prompt_raw, 2)
            datas_for_prompt = [d for d in datas_for_prompt_raw if d not in data_not_used]
            datas_part_2_raw = instruction_type_datas_out.get(k, [])
            if len(datas_part_2_raw) < 2:
                datas_for_prompt.extend(datas_part_2_raw)
            else:
                datas_for_prompt.extend(random.sample(datas_part_2_raw, 2))
            current_prompt_len = len(datas_for_prompt)
            for d_index in range(current_prompt_len, 8):
                datas_for_prompt.append(data_not_used[d_index - current_prompt_len])
            random.shuffle(datas_for_prompt)
            print(datas_for_prompt)
            prompt_self_instruct = prompt_self_instruct_raw
            for i in range(len(datas_for_prompt)):
                prompt_self_instruct += f'Task {i+1}: {datas_for_prompt[i]}\n'
            task_pool.append((k, prompt_self_instruct))
            #print(prompt_self_instruct)
        raw_res = []
        for res in thread_parallel(get_model_res, task_pool, threads=thread_num, name=model.name, extra_paras=(model,)):
            print(res)
            part = []
            if res[0] is not None:
                for v in res[1]:
                    if v != '':
                        part.append((res[0], v, str_is_cjk(v)))
            extra_list = list_rouge_filter(scorer, part)
            raw_res.extend(extra_list)
        raw_res = list_rouge_filter(scorer, raw_res)
        if len(raw_res) == 0:
            empty_times += 1
        instruction_type_datas_in_tmp = copy.deepcopy(instruction_type_datas_out)
        out_fo = os.path.dirname(output_name)
        if not os.path.exists(out_fo):
            os.makedirs(out_fo, exist_ok=True)
        with open(output_name, 'a') as f_out:
            for res in thread_parallel(check_rougeL, raw_res, threads=thread_num, name='check_rougeL', extra_paras=(scorer, instruction_type_datas_in, instruction_type_datas_in_tmp)):
                if res[0] is not None:
                    k, v = res
                    if k not in instruction_type_datas_out:
                        instruction_type_datas_out[k] = []
                    instruction_type_datas_out[k].append(v)
                    out_data = {'instruct_group': k, 'prompt': v}
                    f_out.write(json.dumps(out_data, ensure_ascii=False) + '\n')
                    f_out.flush()
                    out_datas_len += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self-instruct to get more similiar qa pairs')
    parser.add_argument('-i', '--input', type=str, help='the input eval file')
    parser.add_argument('-o', '--output', type=str, default='output/self_instruct/out.jsonl', help='the output file to store the final res')
    parser.add_argument('-l', '--limit', type=int, default=600, help='how many samples generated for the specific task')
    parser.add_argument('-m', '--model', type=str, default='gpt4-turbo', help="The seed gen model")
    parser.add_argument('-t', '--thread', type=int, default=50, help='the thread num when excuting the program')
    args=parser.parse_args()

    input_datas = read_data_file(args.input)

    gen_self_instruct_datas(args.model, input_datas, args.output, args.limit, args.thread)