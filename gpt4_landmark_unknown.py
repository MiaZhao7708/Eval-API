import json
import random
from concurrent import futures
from tqdm import tqdm
from session import session
from abc import ABC, abstractmethod
# from utils.crawlers import get_RMB_exchange
from utils.utils import uuid
import sys
sys.path.append('..')
import openai
import copy
import time
import os
import api_manager
from prompts import prompt_landmark_judge,prompt_for_unknown
from PIL import Image
import base64
from io import BytesIO
import os
import threading
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import argparse

log_directory = "/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/evaluation/logs_landmark"
log_file = os.path.join(log_directory, "processing_landmark_unknown.log")
TEST = True
# 确保日志目录存在
os.makedirs(log_directory, exist_ok=True)

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s [%(levelname)s] %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler(log_file),  # 将日志写入到文件
        logging.StreamHandler()  # 同时在控制台输出日志
    ]
)

if not TEST:
    parser = argparse.ArgumentParser(description="Evaluate model with given checkpoint.")
    parser.add_argument("ckpt", type=str, help="Checkpoint to evaluate")
    parser.add_argument("split", type=str, help="Checkpoint to evaluate")
    args = parser.parse_args()
    CKPT = args.ckpt
    SPLIT = args.split
    MYPATH='/cpfs/29f69eb5e2e60f26/user/sft_intern/zhaoyaqi'
    print(f"====== Evaluating with checkpoint: {CKPT} and split: {SPLIT} ======")
    INPUT_FILE = f"{MYPATH}/LLaVA/cogalign/{CKPT}/{SPLIT}/merge_{SPLIT}_2.5k.jsonl"

def filter_input_prompts(dataset, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            processed_data = [json.loads(line) for line in f]
            processed_ids = list(set([item['question_id'] for item in processed_data]))
        return [item for item in dataset if item['question_id'] not in processed_ids]
    return dataset



def thread_parallel(processor, dataset, threads=10, name=None, extra_paras=None):
    with futures.ThreadPoolExecutor(threads) as executor:
        if extra_paras is None:
            process_futures = [executor.submit(processor, data)for data in dataset]
        else:
            process_futures = [executor.submit(processor, data, *extra_paras)for data in dataset]
        for future in tqdm(futures.as_completed(process_futures), desc=name, total=len(dataset)):
            yield future.result()

def process_parallel(processor, dataset, threads=10, name=None, extra_paras=None):
    with futures.ProcessPoolExecutor(threads) as executor:
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

class Model(ABC):
    def __init__(self, server, name, stream=True, timeout=120, temperature=0.3):
        self.server = server
        self.name = name
        self.stream = stream
        self.timeout = timeout
        self.temperature = temperature
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
    
    @property
    def url(self):
        url = f'http://{self.server}'
        if self.stream:
            url += '/generate_stream'
        else:
            url += '/generate'
        return url
    
    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_stream(self, stream):
        self.stream = stream
    
    @abstractmethod
    def call(self, prompt, history=[], system=None, qtype=''):
        return None

    def process_data(self, data, use_model_gen=True):
        try:
            if 'messages' in data or isinstance(data, list): # it's multi turn
                prompt, qtype, history = None, '', []
                # write ans to 'model' key if all use model gen(which data usually don't have this key), else 'content' key(which data usually have)
                insert_key = 'model' if use_model_gen else 'content'
                system = None
                for message in data['messages'] if 'messages' in data else data:
                    if message['role'] == 'user':
                        # if the user speak continuously
                        if prompt is not None:
                            history.append([prompt, None])
                        prompt = message['content']
                        qtype = message.get('qtype', '')
                    elif message['role'] == 'assistant':
                        if insert_key not in message:
                            message[insert_key] = self.call(prompt=prompt, history=history, system=system, qtype=qtype)
                        history.append([prompt, message[insert_key]])
                        prompt = None
                    # currently only support 1 turn system
                    elif message['role'] in {'system', 'user_system'} and system is None:
                        system = message['content']
                # if the role of last turn is 'user', we should generate answer for it
                if prompt is not None:
                    last_turn_answer = {'role': 'assistant', insert_key: self.call(prompt=prompt, history=history, system=system, qtype=qtype)}
                    if 'messages' in data:
                        data['messages'].append(last_turn_answer)
                    else:
                        data.append(last_turn_answer)
            else: # single turn
                if 'model' not in data:
                    data['model'] = self.call(prompt=data['content'], history=[], system=data.get('system', None), qtype=data.get('qtype', ''))
        except Exception as e:
            print(f"{data}发生错误：{e}", flush=True)
        return data

    
# '47.236.144.103:80'   
class openAIModel(Model):
    def __init__(self, server='10.33.133.244:80', name='gpt-4-1106-preview', stream=False, timeout=180, temperature=1.0, json_mode=False):
        super().__init__(server, name, stream, timeout, temperature)
        self.json_mode = json_mode
        self.api_key = api_manager.get_keys('openAI')
        self.input_price, self.output_price = self.init_price()

    @property
    def url(self):
        return f'http://{self.server}/v1/chat'

    def set_api_key(self, api_key):
        self.api_key = api_key
        api_manager.add_key("openAI", api_key)

    def init_price(self):
        dollar = 7.1    #get_RMB_exchange('美元')
        if self.name == 'gpt-4-1106-preview' or self.name == 'gpt-4-1106-vision-preview':
            return dollar * 0.01 / 1000, dollar * 0.03 / 1000
        elif self.name == 'gpt-4':
            return dollar * 0.03 / 1000, dollar * 0.06 / 1000
        elif self.name == 'gpt-4-32k':
            return dollar * 0.06 / 1000, dollar * 0.12 / 1000
        elif self.name == 'gpt-3.5-turbo-1106':
            return dollar * 0.001 / 1000, dollar * 0.002 / 1000
        elif self.name == 'gpt-3.5-turbo-instruct':
            return dollar * 0.0015 / 1000, dollar * 0.002 / 1000
        # default return gpt3.5-turbo cost
        return dollar * 0.001 / 1000, dollar * 0.002 / 1000
    
    def call_details(self, prompt, history=[], system=None, qtype=''):
        
        openai.api_base = self.url
        messages = []
        if system is not None:
            messages.append({'role': 'system', 'content': system})
        for query, answer in history:
            messages.append({'role': 'user', 'content': query})
            if answer:
                messages.append({'role': 'assistant', 'content': answer})
        messages.append({'role': 'user', 'content': prompt})
        data = dict(
                model=self.name,
                messages=messages,
                temperature=self.temperature
            )
        if self.json_mode:
            data['response_format'] = { "type": "json_object" }

        api_key = None
        if self.api_key is not None:
            if isinstance(self.api_key, str):
                api_key = self.api_key
            else:
                api_key = random.choice(self.api_key)
        if api_key:
            openai.api_key = api_key
        else:
            raise 'No aviliable "openAI" api_key provided in the config/api_keys.json! You should provide them in the file'

        ntry = 0
        while ntry < 5:
            try:
                return self.chat_func(data)
            except Exception as e:
                if type(e) not in [openai.error.ServiceUnavailableError, openai.error.RateLimitError, openai.error.APIConnectionError]:
                    print("access openai error, %s" % (e), type(e))
                    return {'status': 'failed', 'text': None}
                ntry += 1
        print('try too much times')
        return {'status': 'failed', 'text': None}
    
    def chat_func(self, data):
        response = openai.Completion.create(**data)
        if 'choices' not in response:
            print("access openai error, status code: %s，errmsg: %s" % (response.status_code, response.text))
            return {'status': 'failed', 'text': None}
        finish_reason = response['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            print(f'Unexpected finish reason! The finish_reason={finish_reason}')
            return {'status': 'unfinish', 'text': None}
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        res = {}
        res['status'] = 'finish'
        res['text'] = response['choices'][0]['message']['content']
        res['usage'] = (prompt_tokens,completion_tokens)
        return res
    
    def call(self, prompt, history=[], system=None, qtype=''):
        response = self.call_details(prompt, history, system, qtype)
        return response['text']


def get_model(model_name, temperature=None, timeout=None):
    global ALL_MODELS
    if 'ALL_MODELS' not in globals():
        with open('./config/models.json') as f_in:
            ALL_MODELS = json.load(f_in)
    if model_name not in ALL_MODELS:
        raise Exception(f"No model named {model_name} in models/models.json")
    model_config = ALL_MODELS[model_name]
    # added by yaqi
    print(f"!!!!!!! api:{model_config['name']} !!!!!!")
    if model_config['type'] == 'openAI':
        return openAIModel(server=model_config.get('server', '47.236.144.103:80'), 
                            name=model_config['name'], 
                            stream=model_config.get('stream', False), 
                            timeout=timeout if timeout else model_config.get('timeout', 120),
                            temperature=temperature if temperature else model_config.get('temperature', 0.2),
                            json_mode=model_config.get('json_mode', True))

########################################################


def process_item(item, history):
    system ='You are an expert in landmark conversation update.'
    prompt = [
        { 
            "type": "text",
            "text": item['prompt']
        },
    ]
    response = model.call_details(prompt, system=system, history=history)
    result = prompt_for_unknown.parse(item, response)
    return result


def count_lines_in_file(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def run_multithread_processing(threads):
    # with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    #     dataset = [json.loads(line) for line in f]
    # cache_file = os.path.dirname(INPUT_FILE) + '/gpt4_eval_2.5k.jsonl'
    # success_num = count_lines_in_file(cache_file)
    # if success_num == 2500:
    #     print(f'== already successfully save to {cache_file}, pass ==')
    #     return None
    # dataset = filter_input_prompts(dataset, cache_file)

    # changed by yaqi
    # cache_file = '/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/landmark/data/cogalign/data_distribution_pk/25k_3k_unknown_try_gpt4o.jsonl'
    cache_file = '/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/landmark/data/cogalign/data_distribution_pk/75k_5k_unknown_gpt4omini.jsonl'

    dataset = json.load(open('/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/landmark/data/cogalign/data_distribution_pk/75k_5k_unknown.json','r'))
    dataset = pre_process(dataset)
    process_count = 0
    failure_count = 0

    # Initialize model and history
    history = []
    global model
    model = get_model('4o-mini')
    logging.info("Logging system initialized.")
    results = []

    with open(cache_file, 'a') as f:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = {executor.submit(process_item, item, history): item for item in dataset}

            for future in tqdm(futures, desc="Processing items", ncols=100):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        process_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    logging.error(f"Error processing item: {e}, item: {futures[future]}")
                    failure_count += 1
                finally:
                    if len(results) >= 64:  # Example: Write every 64 results
                        f.write("\n".join([json.dumps(res, ensure_ascii=False) for res in results]) + '\n')
                        tqdm.write(
                            f"Current progress: {process_count}/{len(dataset)} | "
                            f"Percentage: {process_count / len(dataset) * 100:.2f}% | "
                            f"Failures: {failure_count}"
                        )
                        results.clear()

        # Write remaining results if any
        if results:
            f.write("\n".join([json.dumps(res, ensure_ascii=False) for res in results]) + '\n')
    print(f"========= success saved to {cache_file} =========")


def pre_process(data):
    # TO FIX: image,landmark_name
    grouped_data = {}
    for item in data:
        question_id = item["id"]
        if question_id not in grouped_data:
            grouped_data[question_id] = []
        grouped_data[question_id].append(item)  
    # print(grouped_data)
    # 定义输入
    info_dicts = []
    for question_id, outputs in grouped_data.items():
        info_dict = {}
        prompt = prompt_for_unknown.get_prompt_unknwon(outputs)
        info_dict['prompt'] = prompt
        info_dict['id'] = question_id
        info_dict['image'] = outputs[0]['image']
        info_dict['landmark_name'] = outputs[0]['landmark_name']
        info_dict['conversations'] = outputs[0]['conversations']
        info_dict['landmark_name_gt'] = outputs[0]['landmark_name_gt']

        info_dicts.append(info_dict)
    return info_dicts


def test():
    history = []
    results = []
    global model
    model = get_model('4o-mini')
    dataset = json.load(open('/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/landmark/data/cogalign/data_distribution_pk/75k_5k_unknown.json','r'))[:2]
    dataset = pre_process(dataset)
    # changed by yaqi
    # import random
    # item = random.choice(dataset)
    system='You are an expert in landmark conversation update.'

    for item in tqdm(dataset, desc="Processing"): 
        prompt = [
            { 
                "type": "text",
                "text": item['prompt']
            },
        ]
        response = model.call_details(prompt, system=system, history=history)
        # import pdb;pdb.set_trace()
        results.append(prompt_for_unknown.parse(item, response))
    # print(results)
    # with open('/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/landmark/data/cogalign/data_distribution_pk/25k_3k_unknown_try_gpt4o.jsonl', 'w', encoding='utf-8') as f:
    with open('/cpfs/29f69eb5e2e60f26/user/sft_intern/lilin/landmark/data/cogalign/data_distribution_pk/75k_5k_unknown_try_gpt4omini.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print('完成处理！')    

if __name__=='__main__':
   
    # test()
    run_multithread_processing(threads=16)

   
    
    
    
    
