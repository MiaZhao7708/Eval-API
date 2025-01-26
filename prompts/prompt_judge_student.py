import re, json, copy

PROMPT_JUDGE = """
## 背景
现在你是一个逻辑学大师。你需要依据 <标准答案和判分要点> 来判断学生每道题的得分\n\n 

## 判分流程
1. 仔细理解prompt的问题描述，思考该题目考的是逻辑学中的什么知识 \n
2. 阅读ground_truth，判断该标准答案是否符合题目要求以及是否回答正确 \n
3. 对学生的答案根据<判分要点>进行打分 ＼n

请注意，对于每道题都需要符合以上流程并且逐条、逐步记录下来。

## 判分要点
5分答案：答案完全正确，需要回答的答案正确，同时思路和过程正确，且回答考虑到了各种可能性，考虑全面 \n
4分答案：答案完全正确，但是没有任何过程 \n
3分答案：答案部分错误，思路大部分正确；或者答案正确，但是过程出现明显错误 \n
2分答案：答案错误，且过程大部分错误 \n
1分答案：答案错误，过程和思路全错\n

请注意，对于有遗漏，不完整的回答，最多只能给3分。\n\n 


## 生成格式
Please detail your thought for understanding the prompt, action the right answers and your reason for the score. 
再以"[]"的格式生成分数，比如：
```
- 结合标准答案，这道题回答结果错误，思路全错，因此得1分，得分是[1]
```
\n\n 

## 题目
{question}

## 标准答案: 
{gt_answer}

## 学生回答:
{answer}
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