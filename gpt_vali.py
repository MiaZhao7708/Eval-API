# 本代码将利用GPT4v验证生成/野站的数据
# 主要目标是删除一些第一次回复错误，依赖提供的标准答案（人工提示）再行纠正的问题
import os
import json
from PIL import Image, ImageDraw, ImageOps
import pyarrow.parquet as pq
import dask.dataframe as dd
from tqdm import trange, tqdm
from io import BytesIO
import random
import base64
import sys
import fire
import copy
from tqdm import trange
sys.path.append('..')
sys.path.append('../evaluation')
from evaluation.model import get_model, thread_parallel
import threading
from concurrent import futures

lock = threading.Lock()

def thread_parallel(processor, dataset, threads=10, name=None, extra_paras=None):
    with futures.ThreadPoolExecutor(threads) as executor:
        if extra_paras is None:
            process_futures = [executor.submit(processor, data)for data in dataset]
        else:
            process_futures = [executor.submit(processor, data, *extra_paras)for data in dataset]
        for future in tqdm(futures.as_completed(process_futures), desc=name, total=len(dataset)):
            yield future.result()

def get_base64_image(path):
    image = Image.open(path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_str = img_base64.decode('utf-8')
    return img_str

SYSTEM_TEMPLATE = {
    'wild_zh': '你将扮演一个分析用户和ai助手对话的专家。',
    'wild_en': 'You will play the role of an expert in analyzing conversations between users and AI assistants.',
    'afanti': '你将扮演一个分析用户和ai助手对话的专家。',
    'chart': '你将扮演一个分析用户和ai助手对话的专家。',
    'SciGraphQA': '你将扮演一个分析用户和ai助手对话的专家。',
    'landmark': '你将扮演一个分析用户和ai助手对话的专家。',
    'food': '你将扮演一个分析用户和ai助手对话的专家。'
}
PROMPT_TEMPLATE = {
    'wild_zh': '以下是一段用户和AI助手的对话。你需要仔细阅读对话并告诉我在对话过程中ai助手是否出现了以下问题：\n1. 用户提醒ai助手没有准确理解要求；\n2. ai助手发现之前回答中出现错误；\n3. ai助手拒绝回答用户请求；\n4. ai助手明确提出用户没有给出图片。\n```\n[Conversation]\n```\n{CONVERSATION}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复；\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（yes/no）代表ai助手是否出现了上述情况中的至少一种；\n3. 注意，你在判别的时候不需要考虑ai助手回复内容的正确性；\n4. 如果你无法确定某种情况是否出现，你要默认ai助手没有出现这个问题。】',
    'wild_en': 'The following is a conversation between a user and an AI assistant. You need to carefully read the conversation and tell me if the AI assistant encountered the following problems during the conversation: \n1. The user reminded the AI assistant that they did not accurately understand the requirements; \n2. The AI assistant noticed an error in the previous answer; \n3. The AI assistant refuses to answer user requests; \n4. The AI assistant found an error in the reference answer provided by the user. \n```\n[Conversation]\n```\n{CONVERSATION}\n```\n[Note that you always reply in JSON format. Your output includes the label field and the reason field. The value of the label field (yes/no) represents whether the AI assistant has encountered at least one of the above situations. The reason field provides your judgement basis.]',
    'afanti': '以下是一段用户和AI助手的对话。你需要仔细阅读对话并告诉我在对话过程中ai助手是否出现了以下问题：\n1. ai助手发现之前回答中出现错误；\n2. ai助手拒绝回答用户请求；\n3. ai助手在回复中提到了参考答案；\n4. ai助手明确提出用户没有给出图片。\n```\n[Conversation]\n```\n{CONVERSATION}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（yes/no）代表ai助手是否出现了上述情况中的至少一种；\n3. 注意，你在判别的时候不需要考虑ai助手回复内容的正确性；\n4. 如果你无法确定某种情况是否出现，你要默认ai助手没有出现这个问题。】',
    'chart': '以下是一段用户和AI助手的对话。你需要仔细阅读对话并告诉我在对话过程中ai助手是否出现了以下问题：\n1. ai助手发现之前回答中出现错误；\n2. ai助手拒绝回答用户请求；\n3. ai助手在回复中提到了参考答案；\n4. ai助手直接说明用户没有给出图片。\n5. ai助手给出的答案与标准答案不符合。\n```\n[Conversation]\n```\n{CONVERSATION}\n```\n```\n[Standard Answers]\n```\n{ANSWER}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（yes/no）代表ai助手是否出现了上述情况中的至少一种；\n3. 如果你无法确定某种情况是否出现，你要默认ai助手没有出现这个问题。】',
    'SciGraphQA': '以下是一段用户和AI助手的对话。你需要仔细阅读对话并告诉我在对话过程中ai助手是否出现了以下问题：\n1. ai助手发现之前回答中出现错误；\n2. ai助手拒绝回答用户请求；\n3. ai助手直接说明用户没有给出图片。\n4. ai助手给出的答案不符合用户要求。\n5. ai助手给出的答案不符合图片内容的实际情况。\n```\n[Conversation]\n```\n{CONVERSATION}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（yes/no）代表ai助手是否出现了上述情况中的至少一种；\n3. 如果你无法确定某种情况是否出现，你要默认ai助手没有出现这个问题。】',
    'landmark': '以下是一段用户和AI助手的对话。你需要仔细阅读对话并告诉我在对话过程中ai助手是否出现了以下问题：\n1. ai助手发现之前回答中出现错误；\n2. ai助手拒绝回答用户请求；\n3. ai助手没有在回答的最开始明确给出一个地点或者地标或者实体的具体名称；\n4. ai助手在回答中提到了参考地点或来自用户的提示。\n5. ai助手给出的地点名称与标准答案不同(语言不同但意思相同视作答案相同)。\n```\n[Conversation]\n```\n{CONVERSATION}\n```\n[Standard Answers]\n```\n{ANSWER}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（yes/no）代表ai助手是否出现了上述情况中的至少一种；】',
    'food': '以下是一段用户和AI助手的对话。你需要仔细阅读对话并告诉我在对话过程中ai助手是否出现了以下问题：\n1. ai助手发现之前回答中出现错误；\n2. ai助手拒绝回答用户请求；\n3. ai助手在回答中提到了参考菜名或参考配料或来自用户的提示。\n4. ai助手给出的菜品名称与标准答案不同(语言不同但意思相同视作答案相同)。\n```\n[Conversation]\n```\n{CONVERSATION}\n```\n[Standard Answers]\n```\n{ANSWER}\n```\n【注意，你的回复需要遵循以下要求。\n1. 你总会以json格式进行回复。\n2. 回复内容包括reason字段和label字段。reason字段给出你的判断理由。label字段的值（yes/no）代表ai助手是否出现了上述情况中的至少一种；】',
}

def process_data(data, model, language, file_name, visited_id):
    if data['id'] in visited_id:
        return
    conversation = ''
    for item in data['conversations']:
        conversation += f"{item['from'].replace('human', 'user').replace('gpt', 'assistant')}: {item['value']}\n"
    prompt = PROMPT_TEMPLATE[language].replace('{CONVERSATION}', conversation)
    if language == 'SciGraphQA':
        # chart data
        answer = ''
        for i, res in enumerate(data['old_responses']):
            answer += f'({i+1}) {res}\n'
        answer = answer.strip()
        prompt = prompt.replace('{ANSWER}', answer)
    elif language in ['landmark', 'food']:
        prompt = prompt.replace('{ANSWER}', data['id'])
    system = SYSTEM_TEMPLATE[language]
    if language == 'SciGraphQA':
        prompt = [
            { 
                "type": "text",
                "text": prompt
            },
            { 
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/jpeg;base64,{get_base64_image(os.path.join('/data_train/wuzhiyu/data/mllm', data['image']))}", 
                    "detail": 'high'
                }
            }
        ]
    response = model.call(prompt, system=system, history=[])
    if response is None:
        return
    else:
        data['gpt_filter'] = response
    global lock
    lock.acquire()
    try:
        with open(file_name, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
    finally:
        lock.release()

def check_dataset(data_path, language):
    try:
        data_list = json.load(open(data_path))
    except:
        data_list = list(map(json.loads, open(data_path)))
    if language == 'SciGraphQA':
        model = get_model('gpt4-turbo-vision')
        data_list = sorted(data_list, key=lambda x: len(x['conversations']), reverse=True)
        print(len(data_list[0]['conversations']))
        data_list = data_list[:5000]
    else:
        model = get_model('gpt4-turbo')
    save_path = data_path.split('.')[0] + '_checked.jsonl'
    visited_id = set()
    if os.path.exists(save_path):
        visited_data = list(map(json.loads, open(save_path)))
        for data in visited_data:
            visited_id.add(data['id'])
    ret_data_list = thread_parallel(process_data, data_list, threads=50, extra_paras=(model, language, save_path, visited_id))
    ret_data_list = [item for item in ret_data_list]
    ret_data_list = list(map(json.loads, open(save_path)))
    print(len(ret_data_list))
    ret_data_list = [item for item in ret_data_list if "\"label\": \"no\"" in item['gpt_filter']]
    save_path = data_path.split('.')[0] + '_filtered.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(ret_data_list, f, indent=4, ensure_ascii=False)
    print(len(ret_data_list))

if __name__ == '__main__':
    fire.Fire()