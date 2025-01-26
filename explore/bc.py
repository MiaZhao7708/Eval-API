import requests
import base64
import json
import concurrent.futures
import tqdm
import hashlib
import os
def bc_docker_api(item,url,idx):
    prompt = item['prompt']
    history = item.get('history',[])
    answer = item.get('ans','')
    if answer:
        item['ans'] = answer
        return item,idx
    parameters = {
        'max_new_tokens': 2048,
        'temperature': 0.3,
        'top_k': 5,
        'top_p': 0.85,
        'repetition_penalty': 1.05,
        'do_sample': True
        }
    input_str = f"<C_Q>{prompt}<C_A>"
    response = requests.post(
        url,
        json={'inputs': input_str, 'parameters': parameters},
        stream=True
    )
    # print(response)    
    for byte_line in response.iter_lines():
        byte_line = byte_line.rstrip(b'\n')
        # import pdb;pdb.set_trace()
        if byte_line.startswith(b'data:'):
            data = json.loads(byte_line.decode().lstrip('data:'))
            answer += data['token']['text']
        else:
            data = json.loads(byte_line.decode())[0]

    answer = data['generated_text']

    item['ans'] = answer
    return  item,idx
def generate_hash(input_string, algorithm='sha256'):
    # 根据指定的算法创建哈希对象
    hash_object = hashlib.new(algorithm)
    # 将字符串编码为字节并更新哈希对象
    hash_object.update(input_string.encode('utf-8'))
    # 获取十六进制格式的哈希字符串
    hash_string = hash_object.hexdigest()
    return hash_string
def get_key(item):
    history = item.get('history',[])
    return item['prompt'] + ''.join(history)
def get_model_result(prompts=[],cache_id='0',url = ''):
    '''
    prompts = [{'prompt':xxx,'history':[],'ans':''},{'prompt':xxx,'history':[],'ans':''}]
    '''
    cache = {}
    cache_file = cache_id+'.json'
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor, \
         tqdm.tqdm(total=len(prompts), ncols=0, desc="get model results") as pbar:
        process_futures = [
            executor.submit(bc_docker_api, prompt,url,idx)
            for idx,prompt in enumerate(prompts)
        ]
        failure_count = 0
        for future in concurrent.futures.as_completed(process_futures):
            result,idx = future.result()
            key = get_key(result)
            if key in cache:continue
            prompts[idx] = result
            with open(cache_file,'a+') as fp:
                fp.write(json.dumps({'key':key,'ans':result['ans']},ensure_ascii=False)+'\n')
                fp.flush()
    return prompts
def test():
    prompts = ['你好','你是谁','1+1']
    inputs = [{'prompt':item} for item in prompts]
    results = get_model_result(prompts=inputs,url = 'http://10.64.8.101:80/generate_stream')
    for item in results:print(item)

test()
