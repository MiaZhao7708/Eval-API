import re, json, copy

import json
import random
import concurrent.futures as cf
import tqdm

PROMPT_JUDGE = """你是一个精通数学、逻辑、各项通识百科的专家，你需要对一个指令的学生回答进行分析，仔细地寻找学生的回答是否有错误和无关信息。评分基础分数为5分。如果学生的回答没有错误，则不扣分。而若对以下情况每出现一次，则扣一次分数。多个情况的扣分可以叠加。若学生最终分数小于1分，则学生最终分数为1分。
- 若学生回答出现了和指令毫不相关的冗余信息，则扣1分。
- 若学生出现了错字错标点等情况，则扣1分。
- 若学生回答步骤中出现了逻辑或事实错误，则扣2分。
- 若学生给的最终答案出现了逻辑或事实错误，则扣2分。
- 若指令要求合理且学生回答不遵循指令要求，则扣2分。
- 若学生的答案出现色情暴力、仇恨辱骂、隐私泄漏、违反法律等违规信息，则扣4分

你需要在分析的最后按照“因此学生答案评分为：x分”(x为你的对应评分)来输出你的评分分数。目标指令和对应的学生回答如下：

目标指令：
```
{prompt}
```

学生回答：
```
{response}
```
###你的分析###
"""

def parse(info_dict, response):
    if response is not None:
        try:
            info_dict['reason'] = response
            res = re.findall('分.*[为是][:：]*\s*([\d\.]+)', response)
            if len(res) == 0:
                res = re.findall('[为是打分][:：]*\s*([\d\.]+)分', response)
            return int(res[-1])
        except:
            print(f'len(res) = {len(res)}, parse failed for response: {response}')
    return None