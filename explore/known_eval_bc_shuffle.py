import json
from tqdm import tqdm
import os
import requests
import json
import concurrent.futures
import hashlib
import os
import random
# 从命令行参数获取CKPT和SPLIT
# CKPT = sys.argv[1]
# SPLIT = sys.argv[2]

CKPT="llava-v1.5-7b-finetune-model4-100"
SPLIT = "landmark_questions"
print(f"ckpt:{CKPT};split:{SPLIT}")

LANGUAGE = SPLIT.split('_')[0]
correct_answers_file = f"/data2/sft/InternData/zhaoyaqi/open-source/LLaVA-Finetune/{LANGUAGE}_en.jsonl"
input_file = f"/data2/sft/InternData/zhaoyaqi/LLaVA/playground/data/eval/world_knowledge/answers/{SPLIT}/{CKPT}/merge.jsonl"
output_file = f"./{SPLIT}/{CKPT}"
if not os.path.exists(output_file):
    os.makedirs(output_file)


PROMPT_TEMPLATE_EN = {
    'landmark': ("Task: Determine if the AI assistant's response matches the 'known' or 'unknown' category based on the provided conversation examples. Each conversation example consists of one question posed by a user, five responses provided by an AI assistant and an unique question_id which belong to the question, all in English. The question_id is used to uniquely identify and reference each question. The 'known' category implies that at least one of the AI assistant's five responses must explicitly mention the specific name of the landmark shown in the picture, similar to the standard answer. The 'unknown' category means that all of the AI assistant's responses fail to mention the specific name of the landmark and only provide generic descriptions or incorrect names.\n"
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
    "[Model:]\n"
    "{'reason': \"All the responses of the AI assistant fail to mention the specific name 'Catedral Metropolitana N' of the landmark shown in the picture. Instead, the responses only provide a generic description of the landmark without specifying its name.\", 'label': 'unknown','question_id': 8247}\n"
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
    "[Model:]\n"
    "{\"reason\": \"All of the responses from AI assistant explicitly mention 'Empire State Building', which matches the specific name given in the standard answer. Therefore, the responses meet the criteria for the 'known' category as they correctly identify the landmark shown in the picture.\", 'label': 'known','question_id':1024}\n"
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
    "[Model:]\n"
    "{'reason': \"The AI assistant mentioned 'stir fried beef with vegetables' and 'stir fried beef with vegetables and spices,' which are similar in dish type to the standard answers 'chinese mongolian beef' and 'scallion fried beef.' Both involve stir fried beef, suggesting a similarity in category despite different specific names or ingredients.\", 'label': 'known','question_id': 791}\n"
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
    "[Model:]\n"
    "{\"reason\": \"The AI assistant's answers all revolve around beef dishes, which are completely different from the standard answer 'vegetarian lasagna,' a dish that does not include meat. Therefore, the responses are not in the same category or even remotely similar to the standard answer.\", 'label': 'unknown','question_id':102}\n"
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
    "[Model:]\n"
    )
}
 
def bc_docker_api(item,url):
    prompt = item['prompt']
    history = item.get('history',[])
    answer = item.get('ans','')
    
    if answer:
        item['ans'] = answer
        return item
    
    parameters = {
        'max_new_tokens': 1024,
        'temperature': 0.3,
        'top_k': 5,
        'top_p': 0.85,
        'repetition_penalty': 1.05,
        'do_sample': True
        }
    
    input_str = f"<C_Q>{prompt}<C_A>"
    
    response = requests.post(
        url,
        headers = {'Content-Type': 'application/json'},
        json={'inputs': input_str, 'parameters': parameters},
        stream=True
    )   
    for byte_line in response.iter_lines():
        byte_line = byte_line.rstrip(b'\n')
        if byte_line.startswith(b'data:'):
            data = json.loads(byte_line.decode().lstrip('data:'))
            answer += data['token']['text']
    answer = data['generated_text']
    item['ans'] = answer
    return item

def generate_hash(input_string, algorithm='sha256'):
    hash_object = hashlib.new(algorithm)
    hash_object.update(input_string.encode('utf-8'))
    hash_string = hash_object.hexdigest()
    return hash_string

def get_key(item):
    history = item.get('history',[])
    return item['prompt'] + ''.join(history)


# load data
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

with open(correct_answers_file, 'r', encoding='utf-8') as f:
    correct_answers = [json.loads(line) for line in f]

correct_answers_dict = {item['my_id']: item['id'] for item in correct_answers}

# process data
language = LANGUAGE
grouped_data = {}
for item in data:
    question_id = item["question_id"]
    if question_id not in grouped_data:
        grouped_data[question_id] = []
    grouped_data[question_id].append(item)  # 同一个question_id有5个answers

prompts = []
paths = []
for question_id, outputs in grouped_data.items():
    question = outputs[0]['prompt']
    answers = {}
    if language == 'food':
        if question_id == 791:
            continue
    if language == 'landmark':
        if question_id == 8247:
            continue
    for i in range(5):
        answers[f"answer_{i+1}"] = outputs[i]['text']
    label = correct_answers_dict.get(question_id, "unknown")
    if label == 'unknown':
        continue
    prompt = PROMPT_TEMPLATE_EN[language].replace('{question}', question).replace('{LABEL}', label).replace('{question_id}', str(question_id))
    for tmp in answers.keys():
        prompt = prompt.replace(tmp, answers[tmp])
    path_tmp = output_file+'/'+str(question_id)+'.json'
    prompts.append(prompt)
    paths.append(path_tmp)


cc = list(zip(prompts, paths))
random.shuffle(cc)
prompts[:], paths[:] = zip(*cc)
inputs = [{'prompt': item} for item in prompts]

for idx,prompt in tqdm(enumerate(prompts)):
    path_save = paths[idx]
    if os.path.exists(path_save):
        continue
    inputs = {'prompt': prompt}
    results = bc_docker_api(inputs,'http://bc4-chat-0628.gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com/generate_stream')
    key = get_key(results)
    item = {'ans': results['ans'], 'key': key}
    with open(path_save,'w') as f:
        json.dump(item, f, ensure_ascii=False, indent=4)