import re, json, copy

PROMPT_JUDGE  = """\
你是精通数学&逻辑推理的专家，需要依据问题的参考答案，对学生作业进行批改，判断学生回答是否正确。打分的评分标准为：
5分答案：满分答案，答案与参考答案一致的同时过程正确，且回答考虑到了各种可能性，考虑全面，可能回答甚至好于参考答案
4分答案：答案与参考答案一致，但是没有过程 \n
3分答案：答案与参考答案不一致，过程大部分正确；或者答案正确，但是过程出现明显错误 \n
2分答案：答案与参考答案不一致，且过程大部分错误 \n
1分答案：答案与参考答案不一致，过程和思路全错\n\n 
你的回答必须严格按照`...，因此得分为[分数]`的格式生成，其中"..."指代评分理由，示例如下：
```
这道题回答与参考答案不一致，但过程大部分正确，因此得分为[3]
```

具体问题为：{prompt}
参考答案为：{label}
学生回答：{response}
"""

def parse(info_dict, response):
    if response is not None:
        info_dict['reason'] = response
        scores = re.findall(r'\[([0-5])\]',response)
        if len(scores) > 0:
            try:
                return int(scores[-1])
            except Exception:
                return None
    return None