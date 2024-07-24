from flask import Flask, request, send_from_directory,send_file,jsonify
import os
import threading
from loguru import logger
import time
import re
import concurrent.futures
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,default='/root/test/mnt')
parser.add_argument('--max_workers',type=int,default=30)
args = parser.parse_args()
files_and_dirs = os.listdir(args.path)
logger.info(f"{args,files_and_dirs}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)

flag = False
app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
tokenizer = None
config = None

def load_model():
    global model,device,flag,tokenizer,config
    tokenizer = AutoTokenizer.from_pretrained(
        args.path, trust_remote_code=True
    )
    #tokenizer.pad_token_id = tokenizer.eod_id
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        trust_remote_code=True
    ).to(device).eval()
    #config = GenerationConfig.from_pretrained(
    #    args.path, trust_remote_code=True, resume_download=True,
    #)
    flag=True
    logger.info("model load complete")

t=threading.Thread(target=load_model)
t.start()

@app.route("/ready", methods=["GET"])
def check_ready():
    global flag
    if not flag:
        return {"status_code": 400},400
    return {"status_code": 200},200
    
@app.route('/v1/translate', methods=['POST'])
def synthesize_speech():
    global model,device,tokenizer,config
    resp = request.get_json()['parameter']
    from_lang = resp['from']
    to_lang = resp['to']
    texts = resp['text']
    logger.info(f"{from_lang,to_lang}")
    all_response = []
    for paragraph in texts:
        prompt = "你是一个日英翻译，请把下面的日文翻译为英文，请按照分隔符翻译，并且只输出译文："+paragraph
        messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ]
        text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        #logger.info(f"{text}")
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        #logger.info(f"{model_inputs}")
        generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
                )
        #logger.info(f"{generated_ids}")
        generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
        #logger.info(f"{generated_ids}")
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #logger.info(f"{response}")
        all_response.append(response)
    return all_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)