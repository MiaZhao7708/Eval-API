import re, json, copy

import json
import random
import concurrent.futures as cf
import tqdm

PROMPT_JUDGE = """指令=【{prompt}】请根据{{语言1, 语言2, 指令约束, 难度, 能力, 属性, 领域}}这七个方面对以上指令进行分析："""

def parse(info_dict, response):
    if response is not None:
        try:
            info_dict['reason'] = response
            columns = ['语言1', '语言2', '指令约束', '难度', '能力', '属性', '领域']
            categorys = []
            for i, part in enumerate(response.split("，")):
                category = part.split('是', 1)[-1].strip('。')
                if i == 0:
                    categorys.extend(category.split('和'))
                else:
                    categorys.append(category)
            assert len(categorys) == len(columns)
            return {k: v for k, v in zip(columns, categorys)}
        except:
            print(f'parse failed for response: {response}')
            print(f'info_dict is {json.dumps(info_dict, ensure_ascii=False)}')
    return None