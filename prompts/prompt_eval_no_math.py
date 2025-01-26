import re, json, copy

PROMPT = '''
你现在是一位全领域专家，需要对一个指令的学生答案进行评判。判断答案是否正确。你的评判逻辑为：
1. 首先对指令本身进行评判。判断指令本身是否是一个有效指令，即指令目标清晰，无遗漏信息，学生可以根据指令给出一个明确的答复
2. 判断学生的答案是否遵循指令，准确地完成了指令的要求，最终结论是否正确，并尽可能不包含无关的冗余信息。在客观事实问题下答案是否专业、正确，是否符合客观事实，是否无细节错误。段落组织是否清晰，完整。答案是否未出现的色情暴力、仇恨辱骂、隐私泄漏等违规信息
3. 判断正确与否时，首先根据第2点，给出正确性的评分理由，随后，给出答案正确性的整体结论。如果指令无效，则正确性为"unknown"。如果答案正确，则正确性为"yes"。如果答案正确性不佳，则正确性为"no"
4. 如果【答案正确性】为"yes"或"unknown"，则【修改结果】为"无"，如果答案正确性为"no"，则你需要给出对于此指令正确的回复

输出的格式（请你严格按照以下格式输出）：
【指令评判】:
【答案评判理由】:
【答案正确性】:
【修改结果】:


以下是1个示例：
----
指令：
```
再精简一些，不超过100字
```

学生回答：
```
我是一位AI助手，擅长回答各种问题。请随时提问，我会尽力提供帮助
```

输出内容：
【指令评判】: 指令并未提到所要精简的目标，学生无法根据指令信息给出一个有效的答复。因此这是一个无效指令
【答案评判理由】: 由于指令无效，所以答案正确性未知
【答案正确性】: unknown
【修改结果】: 无
----

以下是你需要判断的指令和答案：

指令：
```
{prompt}
```

学生回答：

```
{response}
```
'''

def get_res(raw_info, msg):
    if msg + ':' in raw_info:
        part_split = raw_info.split(msg + ':')
    else:
        part_split = raw_info.split(msg + '：')
    return part_split

def parse(info_dict, response):
    data = copy.deepcopy(info_dict['data'])
    if response is not None:
        data['raw_res'] = response
        try:
            part_info = get_res(response, '【修改结果】')
            part, rewrite = part_info[0], part_info[1]
            part_info = get_res(part, '【答案正确性】')
            part, truth = part_info[0], part_info[1]
            part_info = get_res(part, '【答案评判理由】')
            instruct_reason, answer_reason = part_info[0], part_info[1]
            data['instruct_reason'] = instruct_reason.strip()
            data['answer_reason'] = answer_reason.strip()
            data['truth'] = truth.strip()
            data['rewrite'] = rewrite.strip()
            return data
        except Exception as e:
            print('parse error for request={}, response={}, error={}'.format(data['prompt'], response, e))
            return None
    return None