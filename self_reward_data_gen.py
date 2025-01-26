from self_instruct import gen_self_instruct_datas
import argparse
import json5
import os
from glob import glob
from utils.utils import read_data_file
from model import process_parallel

def read_and_parse(data_file, base_dir):
    in_datas = read_data_file(data_file)
    out_datas = []
    for d in in_datas:
        out_d = {}
        turn = 0
        for mes in d['messages']:
            if mes['role'] in {'user', 'assistant'}:
                turn += 1
                if mes['role'] == 'user':
                    out_d['prompt'] = mes['content']
        # don't treat turns larger than 2
        if turn != 2:
            continue
        out_d['instruct_group_list'] = [data_file.split(base_dir, 1)[-1]]
        if out_d['prompt'] != '':
            out_datas.append(out_d)
    return out_datas
    

def file_task(task_pack, model, thread):
    datas, out_f, sample_rate = task_pack
    try:
        sample_num = min(len(datas), int(len(datas)/10*sample_rate))
        gen_self_instruct_datas(model, datas, out_f, sample_num, thread)
        return True
    except Exception as e:
        print(e)
        return False


def produce_self_instruct_datas(base_dir, input_fos, base_output_fo, model, thread):
    tasks = []
    for d_fo, rate in input_fos.items():
        data_fo = os.path.join(base_dir, d_fo)
        if os.path.isfile(data_fo):
            out_f = base_output_fo + '/' + d_fo
            datas = read_and_parse(data_fo, d_fo)
            if len(datas) > 10:
                tasks.append((datas, out_f, rate))
        elif os.path.isdir(data_fo):
            for f in glob(data_fo + '/*.jsonl', recursive=True):
                out_f = base_output_fo + '/' + f.split(base_dir, 1)[-1]
                datas = read_and_parse(f, base_dir)
                if len(datas) > 10:
                    tasks.append((datas, out_f, rate))
        else:
            for f in glob(data_fo + '*.jsonl', recursive=True):
                out_f = base_output_fo + '/' + f.split(base_dir, 1)[-1]
                datas = read_and_parse(f, base_dir)
                if len(datas) > 10:
                    tasks.append((datas, out_f, rate))
    failed = 0
    call_status = process_parallel(file_task, tasks, threads=8, extra_paras=(model, thread))
    for status in call_status:
        if not status:
            failed += 1
    print(failed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='self-instruct to get more similiar training pairs')
    parser.add_argument('-i', '--input', type=str, help='the input config file')
    parser.add_argument('-o', '--output', type=str, default='output/self_reward_instruct', help='the output folder to store the final res')
    parser.add_argument('-m', '--model', type=str, default='gpt4-turbo', help="The seed gen model")
    parser.add_argument('-t', '--thread', type=int, default=40, help='the thread num when excuting the program')
    args=parser.parse_args()

    with open(args.input) as f_in:
        input_config = json5.load(f_in)
    input_fos = input_config['train_files']['sample_rate']
    base_dir = os.path.dirname(args.input)
    produce_self_instruct_datas(base_dir, input_fos, args.output, args.model, args.thread)