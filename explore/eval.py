import json
from tqdm import tqdm
import os
import argparse
import random
import sys
sys.path.append('..')
sys.path.append('../evaluation')
from evaluation.model import get_model

# 定义全局变量
CKPT = "llava-v1.5-7b-pretrain-base-model"
BASE_PATH = "/data2/sft/InternData/zhaoyaqi"
SPLIT = "food_questions"
LANGUAGE = 'food'

# 使用全局变量构建路径
correct_answers_file = f"{BASE_PATH}/open-source/LLaVA-Finetune/{LANGUAGE}_en.jsonl"
input_file = f"{BASE_PATH}/LLaVA/playground/data/eval/world_knowledge/answers/{SPLIT}/{CKPT}/merge.jsonl"
output_file = f"{BASE_PATH}/LLaVA/playground/data/eval/world_knowledge/answers/{SPLIT}/{CKPT}/evaluation_results"

# prompt
SYSTEM_TEMPLATE = {
    'en': 'You will play the role of an expert in analyzing conversations between users and AI assistants.',
    'cn': '你将扮演一个分析用户和ai助手对话的专家。'
}

PROMPT_TEMPLATE = {
    'landmark': '以下是一段用户和AI助手的对话，对话中包括1个问题和相应的5个答案，你需要仔细阅读对话并告诉我在对话过程中ai助手属于哪种情况：\nknown: ai助手的5次回答中至少有1次提到过参考地点或来自用户的提示。\nunknown: ai助手的5次回答中提到的地点名称与标准答案都完全不同(语言不同但意思相同视作答案相同)。\n```\n[user:]\n```\n{question}\n```\n[Assistant:]\n```\n[{answer_1},\n{answer_2},\n{answer_3},\n{answer_4},\n{answer_5}]\n```\n[Standard Answers]\n```\n{LABEL}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（known/unknown）代表ai助手是否出现了上述情况中的某一种；】',
    'food': '以下是一段用户和AI助手的对话，对话中包括1个问题和5个答案，你需要仔细阅读对话并告诉我在对话过程中ai助手属于哪种情况：\nknown: ai助手的5次回答中至少有1次提到过参考菜名或参考配料或来自用户的提示。\nunknown: ai助手的5词回答中提到的菜品名称与标准答案都完全不同(语言不同但意思相同视作答案相同)。\n```\n[user:]\n```\n{question}\n```\n[Assistant:]\n```\n[{answer_1},\n{answer_2},\n{answer_3},\n{answer_4},\n{answer_5}]\n```\n[Standard Answers]\n```\n{LABEL}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（known/unknown）代表ai助手是否出现了上述情况中的某一种；】',
}

# load data
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
random.shuffle(data)

with open(correct_answers_file, 'r', encoding='utf-8') as f:
    correct_answers = [json.loads(line) for line in f]
correct_answers_dict = {item['my_id']: item['id'] for item in correct_answers}


# process data
language = LANGUAGE
model = get_model('gpt4-turbo')
grouped_data = {}
for item in data:
    question_id = item["question_id"]
    if question_id not in grouped_data:
        grouped_data[question_id] = []
    grouped_data[question_id].append(item)  # 同一个question_id有5个answers

for question_id, outputs in tqdm(grouped_data.items(), desc="Evaluating outputs"):
    file_path = os.path.join(output_file, str(question_id)+".json")
    if os.path.exists(file_path):
        continue
    question = outputs[0]['prompt']
    answers = {}
    for i in range(5):
        answers[f"answer_{i+1}"] = outputs[i]['text']
    label = correct_answers_dict.get(question_id, "unknown")
    if label == 'unknown':
        continue
    prompt = PROMPT_TEMPLATE[language].replace('{question}', question).replace('{LABEL}', label)
    for tmp in answers.keys():
        prompt = prompt.replace(tmp, answers[tmp])
    system = SYSTEM_TEMPLATE['en']
    import pdb;pdb.set_trace()
    response = model.call(prompt, system=system, history=[])
    if response is None:
        print('error: response is none')
        to_write={}
    else:
        to_write = {
            'question_id': question_id,
            'gpt_filter': response
        }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(to_write, f, ensure_ascii=False)
