import json, argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from utils.utils import print_and_write

plt.rcParams['font.sans-serif'] = 'Noto Serif CJK SC'

parser = argparse.ArgumentParser(description='call gpt4 to rank the generated responses')
parser.add_argument('-i', '--input', type=str, help='the input jsonl file')
parser.add_argument('-o', '--output', type=str, default='plot', help='the output folder to store heatmaps')
parser.add_argument('-d', '--detail', action='store_true', help='also store the detailed result')
args=parser.parse_args()

def init_matrix(dims):
    matrix = np.zeros((dims,dims), dtype=int)
    return matrix


def calibrate_summary_score(labels):
    cali_dict = {}
    max_score = -999 
    for ele in labels:
        try:
            score = int(ele['体感校验顺序'] + 0.5)
        except Exception:
            print(ele)
            return
        if max_score < score:
            max_score = score
        cali_dict[score] = None
    cali_var = 1
    for i in range(max_score + 1):
        if i in cali_dict.keys():
            cali_dict[i] = cali_var
            cali_var += 1
    for ele in labels:
        ele['8/2综合分'] = int(6 - ele['体感校验顺序'] + 0.5)
        tmp = int(ele['体感校验顺序'] + 0.5)
        ele['体感校验顺序'] = cali_dict[tmp]


def eval_summary_score(summary_eval_matrix, summary_eval_dict, human_label_details, pred_label_details):
    human_labels = []
    pred_labels = []
    for label in human_label_details:
        human_labels.append(int(label['体感校验顺序']))
    for label in pred_label_details:
        #pred_labels.append(int(6.5 - label['体感校验顺序']))
        try:
            pred_labels.append(6 - (float(label['正确分']) * 0.8 + float(label['语言分'] * 0.2)))
        except Exception:
            print(label)
            return
        #pred_labels.append(float(label['正确分']))
    for i in range(1, len(pred_labels)):
        for j in range(i):
            diff = pred_labels[i] - pred_labels[j]
            diff_true = human_labels[i] - human_labels[j]
            if diff < 0:
                diff = -diff
                diff_true = -diff_true
            if diff_true >= 4.5:
                diff_true = 4
            if diff_true <= -4.5:
                diff_true = -4
            if diff >= 4.5:
                diff = 4
            if diff not in summary_eval_dict.keys():
                summary_eval_dict[diff] = [1, 1] if diff_true > 0 else [0, 1]
            else:
                summary_eval_dict[diff][1] += 1
                summary_eval_dict[diff][0] = summary_eval_dict[diff][0] + 1 if diff_true > 0 else summary_eval_dict[diff][0]
            # idx = 0, 0~0.5, 0.5~1.5, 1.5~2.5, 2.5~3.5, 3.5~4.5
            diff_idx = 0 if diff == 0 else int(diff + 0.5) + 1
            # diff_true idx would split to 2 columns for -0.5-0.5 for more detailed order!
            # idx = -4.5~-3.5, -3.5~-2.5, -2.5~-1.5, -1.5~-0.5, -0.5~0, 0, 0~0.5, 0.5~1.5, 1.5~2.5, 2.5~3.5, 3.5~4.5
            if diff_true == 0:
                diff_true_idx = 5
            elif diff_true > 0:
                diff_true_idx = int(diff_true + 0.5) + 6
            elif diff_true > -0.5:
                diff_true_idx = 4
            else:
                diff_true_idx = int(diff_true + 0.5) + 3
            summary_eval_matrix[diff_idx][diff_true_idx] += 1



def get_matrix_result(output_dir, datas, print_to_console=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    #truth, fluent, safe, summary, order
    eval_matrix = [init_matrix(5), init_matrix(5), init_matrix(2), init_matrix(5), init_matrix(5)]
    summary_eval_matrix = np.zeros((6,11), dtype=int)
    summary_eval_dict = {}

    for idx, data in datas.iterrows():
        human_labels = json.loads(data['human_label'])
        pred_labels = json.loads(data['gpt_label'])
        calibrate_summary_score(pred_labels)
        for idx, label in enumerate(human_labels):
            pred = pred_labels[idx]
            for j, key in enumerate(['正确分', '语言分', '安全分', '8/2综合分', '体感校验顺序']):
                var_T = int(float(label[key]) + 0.5)
                try:
                    var_P = int(float(pred[key]) + 0.5)
                except:
                    continue
                if key != '安全分':
                    var_T -= 1
                    var_P -= 1
                if var_T >= 5:
                    var_T = 4
                if var_P >= 5:
                    var_P = 4
                #sometimes the pred fails and the var is set to -999, we skip this case
                if var_P >= 0:
                    eval_matrix[j][var_T][var_P] += 1
        eval_summary_score(summary_eval_matrix, summary_eval_dict, human_labels, pred_labels)

    # print matrix
    with open(f'{output_dir}/diff_acc_matrix.txt', 'w') as f_diff:
        if print_to_console:
            print('*' * 15 + 'Diff Accurate' + '*' * 15)
        print_and_write('Total Diff Accurate matrix:', file=f_diff, to_console=print_to_console)
        print_and_write(summary_eval_matrix, file=f_diff, to_console=print_to_console)
        total_nums = []
        pos_nums = []
        pos_looses = []
        for i in range(len(summary_eval_matrix) - 1, -1, -1):
            sum_num = np.sum(summary_eval_matrix[i])
            pos_num = np.sum(summary_eval_matrix[i][6:])
            pos_loose = np.sum(summary_eval_matrix[i][5:])
            if len(total_nums) == 0:
                total_nums.append(sum_num)
                pos_nums.append(pos_num)
                pos_looses.append(pos_loose)
            else:
                total_nums.append(sum_num + total_nums[-1])
                pos_nums.append(pos_num + pos_nums[-1])
                pos_looses.append(pos_loose + pos_looses[-1])
        human_scores = np.sum(summary_eval_matrix, axis=0)
        #for i in range(1, len(total_nums)):
        #    print_and_write(f'human recall for diff > {0 if i == 1 else i - 1.5}: {human_recall:.4f}', f_diff, print_to_console)
        for i in range(len(total_nums)):
            sum_num, pos_num, pos_loose = total_nums[i], pos_nums[i], pos_looses[i]
            print_str = f'>= {3.5 - i}'
            if i == 4:
                print_str = '> 0'
            elif i == 5:
                print_str = '= 0'
            j = len(total_nums) - i - 1
            human_recall = 1 if j == 0 else (np.sum(human_scores[5+j:]) + np.sum(human_scores[:6-j])) / total_nums[-1]
            print_and_write(f'gpt4 diff {print_str}, sum_num: {sum_num}, pos_num: {pos_num}, pos_loose: {pos_loose}, acc_strict: {0 if pos_num == 0 else pos_num/sum_num:.4f}, acc_loose: {0 if pos_loose == 0 else pos_loose/sum_num:.4f}, recall: {0 if sum_num == 0 else sum_num/total_nums[-1]:.4f}, human_recall: {human_recall:.4f}', f_diff, print_to_console)

    if print_to_console:
        print('*' * 15 + 'Recall for specific Acc' + '*' * 15)
    score_total_num, score_pos_num = 0, 0
    # recalls for acc 90, 80, 70
    recalls = [0, 0, 0]
    for k, v in sorted(summary_eval_dict.items(), reverse=True):
        score_total_num += v[1]
        score_pos_num += v[0]
        if score_pos_num / score_total_num >= 0.9:
            recalls[0] = int(score_total_num / total_nums[-1] * 10000 + 0.5) / 100
        if score_pos_num / score_total_num >= 0.8:
            recalls[1] = int(score_total_num / total_nums[-1] * 10000 + 0.5) / 100
        if score_pos_num / score_total_num >= 0.7:
            recalls[2] = int(score_total_num / total_nums[-1] * 10000 + 0.5) / 100
    with open(f'{output_dir}/diff_recall.csv', 'w') as f_recall:
        print_and_write(f'Recall@70, {recalls[2]}%', f_recall, print_to_console)
        print_and_write(f'Recall@80, {recalls[1]}%', f_recall, print_to_console)
        print_and_write(f'Recall@90, {recalls[0]}%', f_recall, print_to_console)
    # add human recall
    human_recall_at_1 = int((np.sum(human_scores[6:]) + np.sum(human_scores[:5])) / total_nums[-1] * 10000 + 0.5) / 100
    recalls.append(human_recall_at_1)
    
    if print_to_console:
        print('*' * 15 + 'Diff Accurate' + '*' * 15)     
    for j, key in enumerate(['正确分', '语言分', '安全分', '8/2综合分', '体感校验顺序']):
        eval_res = eval_matrix[j]
        T_sums = np.sum(eval_res, axis = 1) + 1e-12
        R_sums = np.sum(eval_res, axis=0) + 1e-12
        P_ratio = np.divide(eval_res, T_sums.reshape(T_sums.shape[0], 1))
        R_ratio = np.divide(eval_res, R_sums.reshape(1, R_sums.shape[0]))
        acc_score = np.diagonal(P_ratio)
        acc_plus_1_score = np.append(np.diagonal(P_ratio[:-1, 1:]), 0)
        acc_minus_1_score = np.insert(np.diagonal(P_ratio[1:, :-1]), 0, 0)
        acc_approx_1_score = acc_score + acc_plus_1_score + acc_minus_1_score
        recall_score = np.diagonal(R_ratio)
        acc_total = np.trace(eval_res) / np.sum(T_sums)
        acc_approx_1_total = (np.trace(eval_res) + np.trace(eval_res[:-1, 1:]) + np.trace(eval_res[1:, :-1])) / np.sum(T_sums)

        key = key.replace('/', '_')
        if print_to_console:
            print('*' * 7 + key + '*' * 7)  
        with open(f'{output_dir}/{key}_acc.csv', 'w') as f_acc:
            print_and_write(f'Total Acc, {acc_total:.4f}', f_acc, print_to_console)
            print_and_write(f'Approx Acc, {acc_approx_1_total:.4f}', f_acc, print_to_console)
            print_and_write('Score, Acc, Recall, F1, Acc_approx_1', f_acc, print_to_console)
            for idx, score in np.ndenumerate(acc_score):
                r_score = recall_score[idx[0]]
                f1_score = (score * r_score * 2) / (score + r_score + 1e-12)
                approx_acc = acc_approx_1_score[idx[0]]
                print_and_write(f'score@{idx[0] + 1}, {score:.4f}, {r_score:.4f}, {f1_score:.4f}, {approx_acc:.4f}', f_acc, print_to_console)
        
        # print matrix
        with open(f'{output_dir}/{key}_matrix.txt', 'w') as f_matrix:
            print_and_write('Matrix Nums: ', file=f_matrix, to_console=print_to_console)
            print_and_write(eval_res, file=f_matrix, to_console=print_to_console)            
            print_and_write('Matrix Accs: ', file=f_matrix, to_console=print_to_console)
            print_and_write(P_ratio, file=f_matrix, to_console=print_to_console)
        
        if print_to_console:
            print('')

        # plot heatmap
        ax = sns.heatmap(P_ratio, vmin=0, vmax=1, cmap='crest')
        ax.set_xlabel('Pred')
        ax.set_ylabel('Truth')
        plt.title(key)
        plt.savefig(f'{output_dir}/{key}_heat.png')
        plt.clf()
    return recalls
     
     
def get_ele_matrix_result(output_dir, datas, col_name):
    splited_datas = [(k, v) for k, v in datas.groupby(col_name)]
    recall_dict = {}
    for key, datas in splited_datas:
        if key == '':
            key = '__empty_value__'
        recall_dict[key] = get_matrix_result(output_dir + '/' + key, datas, print_to_console=False)
    out_excel = pd.DataFrame.from_dict(recall_dict, orient='index', columns=['90%一致率的召回率', '80%一致率的召回率', '70%一致率的召回率', '人评一致率'])
    out_excel.to_excel(f'{output_dir}/recall.xlsx')


if __name__ == '__main__':
    all_datas = []
    ele_cloumns = ['难度', '能力', '标准答案']
    
    with open(args.input) as f_in:
        for line in f_in.readlines():
            data = json.loads(line)
            if len(data['gpt_label']) > 0:
                data_labels = data['labels'][0]
                eval_data = [data_labels.get(ele, '') for ele in ele_cloumns]
                eval_data.append(json.dumps(data['labels'], ensure_ascii=False))
                eval_data.append(json.dumps(data['gpt_label']))
                all_datas.append(eval_data)
    column_names = ele_cloumns.copy()
    column_names.append('human_label')
    column_names.append('gpt_label')
    df_data = pd.DataFrame(all_datas, columns=column_names)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    get_matrix_result(args.output + '/total', df_data)
    if args.detail:
         for col_name in ele_cloumns:
              get_ele_matrix_result(args.output + '/' + col_name, df_data, col_name)