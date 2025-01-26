import json, argparse
import pandas as pd
import numpy as np
from utils.utils import print_and_write

def cal_stability(df_data, model_list, group_name_list, output_file_name):
    with open(output_file_name, 'w', encoding='utf-8') as f_out:
        for model_name in model_list:
            # scan twice in case we have col_names like ['model_1', 'model_2', 'model_past_1', 'model_past_2']
            col_names = [col for col in df_data if col.startswith(f'score_{model_name}')]
            max_i = 1
            for name in col_names:
                i = int(name.split('_')[-1])
                max_i = i if max_i < i else max_i
            filter_col = [f'score_{model_name}_{i}' for i in range(1, max_i + 1)]
            all_nums = len(df_data)
            passed_rate = np.array([len(df_data.loc[df_data[col] > 3]) / all_nums for col in filter_col])
            avg = np.average(passed_rate)
            std = np.std(passed_rate, ddof=1)
            print_and_write(f'{model_name}, all, {avg:.4f}, {std:.4f}, [{avg - 2 * std:.4f} - {avg + 2 * std:.4f}]', f_out)
            for col_name in group_name_list:
                splited_datas = [(k, v) for k, v in df_data.groupby(col_name)]
                for part_key, part_data in splited_datas:
                    part_nums = len(part_data)
                    passed_part_rate = [len(part_data.loc[part_data[col] > 3]) / part_nums for col in filter_col]
                    avg = np.average(passed_part_rate)
                    std = np.std(passed_part_rate, ddof=1)
                    print_and_write(f'{model_name}, {part_key}, {avg:.4f}, {std:.4f}, [{avg - 2 * std:.4f} - {avg + 2 * std:.4f}]', f_out)

            #print(df_data[filter_col] > 3)
            print_and_write(f'{model_name} pass rate, {passed_rate}', f_out)
            model_avg = np.round(np.average(df_data[filter_col] > 3, axis=1), 6)
            model_stability = np.round(np.std(df_data[filter_col] > 3, ddof=1, axis=1), 6)
            avg_stability = np.average(model_stability)
            std_stability = np.std(model_stability, ddof=1)
            print_and_write(f'{model_name}, stability, {avg_stability:.4f}, {std_stability:.4f}, [{avg_stability - 2 * std_stability:.4f} - {avg_stability + 2 * std_stability:.4f}]', f_out)
            unique, counts = np.unique(model_avg, return_counts=True)
            print_and_write(f'{model_name}, avg_detail, {dict(zip(unique, counts))}', f_out)
            unique, counts = np.unique(model_stability, return_counts=True)
            print_and_write(f'{model_name}, stability_detail, {dict(zip(unique, counts))}', f_out)
            for col_name in group_name_list:
                splited_datas = [(k, v) for k, v in df_data.groupby(col_name)]
                for part_key, part_data in splited_datas:
                    model_avg = np.round(np.average(part_data[filter_col] > 3, axis=1), 6)
                    model_stability = np.round(np.std(part_data[filter_col] > 3, ddof=1, axis=1), 6)
                    avg_stability = np.average(model_stability)
                    std_stability = np.std(model_stability, ddof=1)
                    print_and_write(f'{model_name}, stability_{part_key}, {avg_stability:.4f}, {std_stability:.4f}, [{avg_stability - 2 * std_stability:.4f} - {avg_stability + 2 * std_stability:.4f}]', f_out)
                    unique, counts = np.unique(model_avg, return_counts=True)
                    print_and_write(f'{model_name}, avg_{part_key} detail, {dict(zip(unique, counts))}', f_out)
                    unique, counts = np.unique(model_stability, return_counts=True)
                    print_and_write(f'{model_name}, stability_{part_key} detail, {dict(zip(unique, counts))}', f_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The eval jsonl input')
    parser.add_argument('-i', '--input', type=str, help='the input eval file')
    parser.add_argument('-o', '--output', type=str, default='output/stability_eval/eval_res.csv', help='the output file to store the judge result')
    parser.add_argument('-m', '--models', type=str, default='bc_online_past,bc_online', help="The eval models, split eahc model using comma for multi model eval case")
    args=parser.parse_args()

    input_datas = pd.read_json(args.input, lines=True)
    model_list = args.models.strip().split(',')
    cal_stability(input_datas, model_list=model_list, group_name_list=['难度'], output_file_name=args.output)