import json

in_datas = []

with open('../output/auto_eval/output_gpt4_eval.jsonl') as f_in:
    for line in f_in.readlines():
        in_datas.append(json.loads(line))

for d in in_datas:
    d.pop('Prompt')
    d.pop('标准答案')
    sample_result = []
    ans_set = {}
    for i in range(1,6):
        ans = d.pop(f'ans_bc_online_current_{i}')
        score = d.pop(f'score_bc_online_current_{i}')
        if ans not in ans_set.keys():
            ans_set[ans] = [score]
        else:
            ans_set[ans].append(score)
    for k, v in ans_set.items():
        score = int(sum(v) / len(v) * 10 + 0.5) / 10
        sample_result.append({'ans': k, 'score': score})
    d['sample_results'] = sample_result

with open('../output/auto_eval/gpt4_auto_eval.jsonl', 'w', encoding='utf-8') as f_out:
    json.dump(in_datas, f_out, ensure_ascii=False, indent=2)