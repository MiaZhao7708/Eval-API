import json
from tqdm import tqdm
import os
import argparse
import random
import sys
sys.path.append('..')
sys.path.append('../evaluation')
import requests
import json
import concurrent.futures
# import tqdm
import hashlib
import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Evaluate model with given checkpoint.")
parser.add_argument("ckpt", type=str, help="Checkpoint to evaluate")
args = parser.parse_args()
CKPT = args.ckpt
MYPATH='/cpfs/29f69eb5e2e60f26/user/sft_intern/zhaoyaqi'
# CKPT="llava-v1.5-7b-finetune-base-baseline"
SPLIT = CKPT.split('finetune-')[-1].split('-')[0]
print(f"Evaluating with checkpoint: {CKPT}; split: {SPLIT}")

# LANGUAGE = SPLIT.split('_')[0]
correct_answers_file_food = f"{MYPATH}/open-source/LLaVA-Finetune/food_en.jsonl"
correct_answers_file_landmark = f"{MYPATH}/open-source/LLaVA-Finetune/landmark_en.jsonl"
input_file = f"{MYPATH}/LLaVA/playground/data/eval/world_knowledge/answers/{SPLIT}/{CKPT}/merge.jsonl"
output_file = f"{MYPATH}/LLaVA/playground/data/eval/world_knowledge/answers/{SPLIT}/{CKPT}"

PROMPT_TEMPLATE_EN = {
    'landmark': ("Task: Determine if the AI assistant's response matches the 'known' or 'unknown' category based on the provided conversation examples. Each conversation example consists of one question posed by a user, five responses provided by an AI assistant and an unique question_id which belong to the question, all in English. The question_id and split is used to uniquely identify and reference each question. The 'known' category implies that at least one of the AI assistant's five responses must explicitly mention the specific name of the landmark shown in the picture, similar to the standard answer. The 'unknown' category means that all of the AI assistant's responses fail to mention the specific name of the landmark and only provide generic descriptions or incorrect names.\n"
    "\n"
    "## Example 1:\n"
    "[user:]\n"
    "What is shown in the picture?\nPlease answer the question.\n"
    "\n"
    "[Assistant:]\n"
    "[{The picture shows a church with a clock tower and a statue of a man in front of it.},\n"
    "{The picture shows a church with a clock tower and a statue of a man in front of it.},\n"
    "{The picture shows a church with a clock tower and a statue in front of it.},\n"
    "{The picture shows a church with a clock tower and a statue of a person on the front.},\n"
    "{The picture shows a church in a city.}]\n"
    "\n"
    "[Standard Answers]\n"
    "Catedral_Metropolitana_N\n"
    "\n"
    "[Question ID]\n"
    "8247\n"
    "\n"
    "[SPLIT]\n"
    "landmark\n"
    "\n"
    "[Model:]\n"
    "{'reason': \"All the responses of the AI assistant fail to mention the specific name 'Catedral Metropolitana N' of the landmark shown in the picture. Instead, the responses only provide a generic description of the landmark without specifying its name.\", 'label': 'unknown','question_id': 8247, 'split': 'landmark'}\n"
    "\n" 
    "## Example 2:\n"
    "[user:]\n"
    "What is the name of the building shown in the picture? Please answer the question.\n"
    "\n"
    "[Assistant:]\n"
    "[{The picture is of the Empire State Building located in New York City.},\n"
    "{It's the Empire State Building, a famous skyscraper in New York.},\n"
    "{This is an iconic building in New York, known as the Empire State Building.},\n"
    "{The building shown is the Empire State Building.},\n"
    "{That's a photo of the Empire State Building, a major landmark in NYC.}]\n"
    "\n"
    "[Standard Answers]\n"
    "Empire State Building\n"
    "\n"
    "[Question ID]\n"
    "1024\n"
    "\n"
    "[SPLIT]\n"
    "landmark\n"
    "\n"
    "[Model:]\n"
    "{\"reason\": \"All of the responses from AI assistant explicitly mention 'Empire State Building', which matches the specific name given in the standard answer. Therefore, the responses meet the criteria for the 'known' category as they correctly identify the landmark shown in the picture.\", 'label': 'known','question_id':1024, 'split': 'landmark'}\n"
    "\n"
    "## New Conversation:\n"
    "[user:]\n"
    "{question}\n"
    "\n"
    "[Assistant:]\n"
    "[{answer_1},\n"
    "{answer_2},\n"
    "{answer_3},\n"
    "{answer_4},\n"
    "{answer_5}]\n"
    "\n"
    "[Standard Answers]\n"
    "{LABEL}\n"
    "\n"
    "[Question ID]\n"
    "{question_id}\n"
    "\n"
    "[SPLIT]\n"
    "landmark\n"
    "\n"
    "[Model:]\n"
    ),
    'food': ("Task: Determine if the AI assistant's response matches the 'known' or 'unknown' category based on the provided conversation examples. Each conversation example consists of one question posed by a user, five responses provided by an AI assistant and an unique question_id which belong to the question, all in English. The question_id is used to uniquely identify and reference each question. The 'known' category implies that the AI assistant's responses reference a dish name or ingredient similar to the standard answer or from the user's prompt. The 'unknown' category means that the AI assistant's responses mention dish names completely different from the standard answer.\n"
    "\n"
    "## Example 1:\n"
    "[user:]\n"
    "Identify the name of the dish. Please answer the question.\n"
    "\n"
    "[Assistant:]\n"
    "[{The dish is called stir fried beef with vegetables.},\n"
    "{The dish is called chicken curry.},\n"
    "{The dish is called stir fried beef with vegetables and spices.},\n"
    "{The dish is called chicken curry.},\n"
    "{The dish in the image is called chicken curry.}]\n"
    "\n"
    "[Standard Answers]\n"
    "chinese mongolian beef, scallion fried beef\n"
    "\n"
    "[Question ID]\n"
    "791\n"
    "\n"
    "[SPLIT]\n"
    "food\n"
    "\n"
    "[Model:]\n"
    "{'reason': \"The AI assistant mentioned 'stir fried beef with vegetables' and 'stir fried beef with vegetables and spices,' which are similar in dish type to the standard answers 'chinese mongolian beef' and 'scallion fried beef.' Both involve stir fried beef, suggesting a similarity in category despite different specific names or ingredients.\", 'label': 'known','question_id': 791, 'split': 'food'}\n"
    "\n"
    "## Example 2:\n"
    "[user:]\n"
    "What dish is this?\n"
    "\n"
    "[Assistant:]\n"
    "[{The dish is called beef stew.},\n"
    "{The dish is known as beef bourguignon.},\n"
    "{It is called beef goulash.},\n"
    "{This is a beef casserole.},\n"
    "{It's a traditional beef roast.}]\n"
    "\n"
    "[Standard Answers]\n"
    "vegetarian lasagna\n"
    "\n"
    "[Question ID]\n"
    "102\n"
    "\n"
    "[SPLIT]\n"
    "food\n"
    "\n"
    "[Model:]\n"
    "{\"reason\": \"The AI assistant's answers all revolve around beef dishes, which are completely different from the standard answer 'vegetarian lasagna,' a dish that does not include meat. Therefore, the responses are not in the same category or even remotely similar to the standard answer.\", 'label': 'unknown','question_id':102, 'split': 'food'}\n"
    "\n"
    "## New Conversation:\n"
    "[user:]\n"
    "{question}\n"
    "\n"
    "[Assistant:]\n"
    "[{answer_1},\n"
    "{answer_2},\n"
    "{answer_3},\n"
    "{answer_4},\n"
    "{answer_5}]\n"
    "\n"
    "[Standard Answers]\n"
    "{LABEL}\n"
    "\n"
    "[Question ID]\n"
    "{question_id}\n"
    "\n"
    "[SPLIT]\n"
    "food\n"
    "\n"
    "[Model:]\n"
    )
}
 
def bc_docker_api(item,url,idx):
    prompt = item['prompt']
    history = item.get('history',[])
    answer = item.get('ans','')
    
    if answer:
        item['ans'] = answer
        return item,idx
    
    parameters = {
        'max_new_tokens': 1024,
        'temperature': 0.3,
        'top_k': 5,
        'top_p': 0.85,
        'repetition_penalty': 1.05,
        'do_sample': True
        }
    
    input_str = f"<C_Q>{prompt}<C_A>"
    try: 
        response = requests.post(
            url,
            json={'inputs': input_str, 'parameters': parameters},

            stream=True,
            timeout=20
        )
    except:
        return None,idx
    # import pdb;pdb.set_trace()
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code} for prompt")
        return None,idx
        # try:
        #     if retries < 10:
        #         print(f"Error: Received status code {response.status_code} for prompt , retry {retries+1} times")
        #         return bc_docker_api(item,url,idx,retries+1)
        #     else:
        #         print(f"Max retries exceeded for prompt")
        #         return None, idx
        # except requests.RequestException as e:
        #     print(f"Request failed for prompt, error: {e}")
        #     return None, idx

    data = None
    for byte_line in response.iter_lines():
        byte_line = byte_line.rstrip(b'\n')
        if byte_line.startswith(b'data:'):
            try:
                data = json.loads(byte_line.decode().lstrip('data:'))
                answer += data['token']['text']
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            data = json.loads(byte_line.decode())[0]
    if data is not None:
        answer = data.get('generated_text', answer)
    else:
        return None,idx 
    
    item['ans'] = answer
    return item, idx

def generate_hash(input_string, algorithm='sha256'):
    hash_object = hashlib.new(algorithm)
    hash_object.update(input_string.encode('utf-8'))
    hash_string = hash_object.hexdigest()
    return hash_string

def get_key(item):
    history = item.get('history',[])
    return item['prompt'] + ''.join(history)

def get_model_result(prompts=[],cache_id='bc_eval',url=''):
    '''
    prompts = [{'prompt':xxx,'history':[],'ans':''},{'prompt':xxx,'history':[],'ans':''}]
    '''
    cache = {}
    cache_file = cache_id+'.json'
    cache_file = os.path.join(output_file,cache_file)
    if os.path.exists(cache_file):
        cache = {}
        for line in open(cache_file):
            js = json.loads(line)
            cache[js['key']] = js['ans']
        
        for idx in range(len(prompts)):
            item = prompts[idx]
            # import pdb;pdb.set_trace()
            key = get_key(item)
            if key in cache:
                prompts[idx]['ans'] = cache[key]

    # urls = ['http://10.64.8.61:80/generate_stream','http://10.64.8.101:80/generate_stream']
    # url = 'http://10.64.24.101:80/generate_stream'
    # random.choice(urls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor, \
         tqdm(total=len(prompts), ncols=0, desc="get model results") as pbar:
        process_futures = [
            executor.submit(bc_docker_api, prompt,url,idx)
            for idx,prompt in enumerate(prompts)
        ]
        for future in concurrent.futures.as_completed(process_futures):
            result,idx = future.result()
            key = get_key(result)
            if key in cache:
                pbar.update(1)
                continue
            if result is None:
                print(f"----- process error, result is None, prompt:{result['prompt']}")
                pbar.update(1)
                continue
            prompts[idx] = result
            with open(cache_file,'a+') as fp:
                fp.write(json.dumps({'ans': result['ans'], 'key': key}, ensure_ascii=False) + '\n')
                fp.flush()
            pbar.update(1)

    return prompts

# load data
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

with open(correct_answers_file_food, 'r', encoding='utf-8') as f:
    correct_answers_food = [json.loads(line) for line in f]

with open(correct_answers_file_landmark, 'r', encoding='utf-8') as f:
    correct_answers_landmark = [json.loads(line) for line in f]


correct_answers_dict_food = {item['my_id']: item['id'] for item in correct_answers_food}
correct_answers_dict_landmark = {item['my_id']: item['id'] for item in correct_answers_landmark}

# 分组数据
grouped_data_food = {}
grouped_data_landmark = {}
for item in data:
    question_id = item["question_id"]
    if item['split'] == 'food':
        if question_id not in grouped_data_food:
            grouped_data_food[question_id] = []
        grouped_data_food[question_id].append(item)  # 同一个question_id有5个answers
    elif item['split'] == 'landmark':
        if question_id not in grouped_data_landmark:
            grouped_data_landmark[question_id] = []
        grouped_data_landmark[question_id].append(item)  # 同一个question_id有5个answers

# 定义存储prompts的列表
prompts = []

# 处理food数据
for question_id, outputs in grouped_data_food.items():
    question = outputs[0]['prompt']
    answers = {}
    language = 'food'
    label = correct_answers_dict_food.get(question_id, "unknown")
    if label == 'unknown':
        continue
    for i in range(5):
        answers[f"answer_{i+1}"] = outputs[i]['text']
    prompt = PROMPT_TEMPLATE_EN[language].replace('{question}', question).replace('{LABEL}', label).replace('{question_id}', str(question_id))
    for tmp in answers.keys():
        prompt = prompt.replace(tmp, answers[tmp])
    prompts.append(prompt)

# 处理landmark数据
for question_id, outputs in grouped_data_landmark.items():
    question = outputs[0]['prompt']
    answers = {}
    language = 'landmark'
    label = correct_answers_dict_landmark.get(question_id, "unknown")
    if label == 'unknown':
        continue
    for i in range(5):
        answers[f"answer_{i+1}"] = outputs[i]['text']
    prompt = PROMPT_TEMPLATE_EN[language].replace('{question}', question).replace('{LABEL}', label).replace('{question_id}', str(question_id))
    for tmp in answers.keys():
        prompt = prompt.replace(tmp, answers[tmp])
    prompts.append(prompt)

# 构建inputs
inputs = [{'prompt': item} for item in prompts]
# import pdb;pdb.set_trace()
# 获取模型结果
results = get_model_result(prompts=inputs, url='http://bc4-chat-0805-rs-v2-200.gw-mfse5p8yujt12gzlrw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com')
