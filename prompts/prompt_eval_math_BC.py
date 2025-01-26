import re, json, copy

import json
import random
import concurrent.futures as cf
import tqdm

PROMPT_JUDGE = """
你是一个数学老师，请阅读题目，结合参考答案认真分析判断学生的回答是否正确;
你的步骤是1、抽取出学生的答案中表示最终答案的表达式，2、把学生答案和参考答案的结果比对，谨慎思考是否一致，3、如果学生作答正确，输出True；否则输出False：\\
需要注意部分答案是等价的，包括但不限于x^2-1=0和x^2=1等价，0.33和1/3等价...\\
请按如下格式作答：
```
学生答案: {{抽取的学生结果表达式}}；判断结果：{{True｜False}}
```
数学题：把5厘米长的线段分成两段，并且以每一段为边长画正方形，己知两个正方形的面积差10平方厘米，大正方形的面积是多少？\\
参考答案：12.25
学生的回答：设大正方形边长为x，小正方形边长就是5-x，那么x^2-(5-x)^2=12.25，解得x=3.5，所以大正方形的面积是3.5^2=12.25平方厘米；\\
你的判断：学生答案：12.25平方厘米；参考答案: 12.25；判断结果：True\\
数学题:{prompt}
参考答案:{prompt}
学生的回答:{response}
你的判断:
"""

def parse(info_dict, response):
    if response is not None:
        info_dict['reason'] = response
        if "True" in response:
            return 1
        else:
            return 0
    return None