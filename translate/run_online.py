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
from openai import OpenAI
executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)

flag = False
app = Flask(__name__)
device = None
model = None
tokenizer = None
config = None
client = None

def load_model():
    global model,device,flag,tokenizer,config,client
    client = OpenAI(api_key = "c6245850a2224ab084a266145102e35f", 
                base_url="http://modelhub.4pd.io/learnware/models/openai/4pd/api/v1")
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        args.path, trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        trust_remote_code=True
    ).to(device).eval()
    config = GenerationConfig.from_pretrained(
        args.path, trust_remote_code=True, resume_download=True,
    )
    '''
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
    global model,device,tokenizer,config,client
    resp = request.get_json()['parameter']
    from_lang = resp['from']
    to_lang = resp['to']
    texts = resp['text']
    logger.info(f"{from_lang,to_lang}")
    all_response = []
    for paragraph in texts:
        prompt = "你是一个日英翻译，请把下面的日文翻译为英文，请按照分隔符翻译，并且只输出译文："+paragraph
        res = client.chat.completions.create(
            model="public/qwen2-72b-instruct-gptq-int4@main", 
            messages=[{ "role": "user", "content": prompt }],
            temperature=1, 
            max_tokens=128, 
            top_p=1, 
            stop=None, 
            )
        all_response.append(res.choices[0].message.content)
    return all_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)