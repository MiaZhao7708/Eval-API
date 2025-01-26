import json
from tqdm import tqdm
import os
import random
import sys
sys.path.append('..')
sys.path.append('../evaluation')
import requests
import json
import concurrent.futures
import hashlib
import os
import sys

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

def get_model_result(prompts=[],cache_id='bc_eval',url = ''):
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
            key = get_key(item)
            if key in cache:
                prompts[idx]['ans'] = cache[key]

    # urls = ['http://10.64.8.61:80/generate_stream','http://10.64.8.101:80/generate_stream']
    # url = 'http://10.64.24.101:80/generate_stream'
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

def get_prompt(data):
    label_id = data["label_id"]
    answer = data["ans"]
    qid = data["id"]
    question = f"Please describe {label_id}. What is it, where is it located, and why is it significant? Can you also mention some notable features and other famous landmarks nearby?"
    response = answer

    prompt = f"The following is a question and its response from an AI model. Evaluate whether the response accurately describes the landmark {label_id} in terms of its location, significance, notable features, and nearby famous landmarks. Provide a judgment of 'known' or 'unknown' and justify your reasoning. Attention: respond in the JSON format as follows: {{\"id\": {qid}, \"result\": \"known\" or \"unknown\", \"reason\": \"your justification\"}}.\n\nQuestion: {question}\n\nResponse: {response}\n\nEvaluation:"
    
    return prompt

# prompts
prompts = []
output_file = '/data2/sft/InternData/zhaoyaqi/LLaVA/explore/llm_result/'
input_file = '/data2/sft/InternData/zhaoyaqi/LLaVA/explore/llm_result/merge.json'
data = json.load(open(os.path.join(input_file),'r'))
for item in data:
    prompt = get_prompt(item)
    prompts.append(prompt)
inputs = [{'prompt': item} for item in prompts]
# import pdb;pdb.set_trace()
results = get_model_result(prompts=inputs, url='http://10.64.8.101:80/generate_stream')
