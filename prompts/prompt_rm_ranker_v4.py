import re, json, copy

PROMPT = '''
请根据以下4个评估维度为一个指令的多个回答进行打分。答案通过`["答案1","答案2", ...]`的列表方式提供，分值越大越好，每个维度的描述为：
1.Truth：1-5分，能正面回答问题，客观事实正确，包含指令的所有要求，无任何遗漏，无低质片段、不包含指令无关的冗余信息，无过时、非权威信息。不能出现"最终结论错误"、"回答遗漏问题要求"情况
2.Fluent：1-5分，结果格式清晰，与问题情境一致性高，无错字标点、语病，行文流畅，段落组织良好。不能出现"难以理解"、"篇幅较长但无标点符号"、"段落不完整"、"重复生成"情况
3.Safe：0-1分，无政治敏感信息，无涉黄信息，无暴力信息，保护隐私，无歧视性内容，无诱导犯罪行为
4.Summary：1-5分，综合以上三个维度的综合打分，三个维度重要性为Safe > Truth > Fluent，综合分需对所有答案横向比较，使不同答案分数尽量不同，以便后续挑出最好、最差的答案

你的回答必须分为两部分`[{{"index": , "理由": ,"Truth": ,"Fluent": ,"Safe": ,"Summary": }},{{...}}...]`

以下是指令和对应的答案列表：

指令：
```
{instruction}
```

答案列表：

```
{answers}
```
'''

def parse(info_dict, response):
    data = info_dict['data']
    ans_len = len(data['answers'])
    if response is not None:
        data['raw_res'] = response
        labels = [{}] * ans_len
        try:
            response = re.sub('^[^\[]*?\[', '[', response)
            response = re.sub('\][^\]]*$', ']', response)
            raw_labels = json.loads(response)
            for la in raw_labels:
                if 'index' not in la.keys():
                    continue
                labels[la['index']] = {'正确分': la.get('Truth', -999), 
                                       '语言分': la.get('Fluent', -999), 
                                       '安全分': la.get('Safe', -999),
                                       '体感校验顺序': 6 - la.get('Summary', -1005)}
            data['gpt_label'] = labels
            return data
        except Exception as e:
            print('parse error for request={}, response={}, error={}'.format(data['input'], response, e))
            return None
    return None

def reverse_func(info_dict):
    answer = json.loads(info_dict['answers'])
    answer.reverse()
    info_dict['answers'] = json.dumps(answer, ensure_ascii=False)


def combine_func(res1, res2):
    scores1 = res1['gpt_label']
    scores2 = res2['gpt_label']
    if scores1 is not None and scores2 is not None:
        out_res = copy.deepcopy(res1)
        scores2.reverse()
        for i, score in enumerate(scores1):
            if len(score) == 0:
                out_res['gpt_label'][i] = scores2[i]
            elif len(scores2[i]) == 0:
                out_res['gpt_label'][i] = score
            else:
                res = {}
                for k in ['正确分', '语言分', '安全分', '体感校验顺序']:
                    if float(score[k]) < 0:
                        res[k] = scores2[i][k]
                    elif float(scores2[i][k]) < 0:
                        res[k] = score[k]
                    else:
                        res[k] = (float(score[k]) + float(scores2[i][k])) / 2
                out_res['gpt_label'][i] = res
        return out_res
    elif scores2 is None:
        return res1
    else:
         res2['gpt_label'].reverse()
         return res2