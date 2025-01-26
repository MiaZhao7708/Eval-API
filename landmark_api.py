from PIL import Image
import base64
from io import BytesIO
import json
import random
from concurrent import futures
from tqdm import tqdm
from session import session
from abc import ABC, abstractmethod
from utils.utils import uuid
import sys
sys.path.append('..')
import openai
import copy
import time
import os
import api_manager
from model import get_model

def get_base64_image(path):
    image = Image.open(path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes)
    img_str = img_base64.decode('utf-8')
    return img_str

model = get_model('gpt4o')
history = []
MYPATH = '/cpfs/29f69eb5e2e60f26/user/sft_intern/zhaoyaqi'
image_folder = f'{MYPATH}/open-source/LLaVA-Finetune/world_knowledge/google-v2/train'

# get data
input_path = f'{MYPATH}/open-source/LLaVA-Finetune/world_knowledge/google-v2/train_select_one.json'
data = json.load(open(input_path,'r'))
for item in data:
    image_path = os.path.join(image_folder,item['image'])
    prompt = [
        { 
            "type": "text",
            "text": input('请输入prompt:')
        },
        { 
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{get_base64_image(image_path)}", 
                'detail': 'high'
            } 
        }
    ]
    response = model.call(prompt, system=None, history=history)
    print('response:', response)
    history.append((prompt, history))