from flask import Flask, request, Response, jsonify, stream_with_context, abort, copy_current_request_context
import requests
import os
import json
import logging
from typing import Dict
import threading
import re
import argparse##注意这个
from prompt import ZhPrompt as Prompt
from vllm_utils import run_vllm_server
import concurrent.futures
default_language = os.getenv('lang', 'german')
default_max_workers = os.getenv('workers', 5)
print(f"default language:{default_language}")
print(f"default workers:{default_max_workers}")
logging.info(f"default language:{default_language}")
logging.info(f"default workers:{default_max_workers}")
parser = argparse.ArgumentParser()
parser.add_argument('--language',type=str,default=default_language)
parser.add_argument('--max_workers',type=int,default=default_max_workers)
args = parser.parse_args()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='# 庄沁宇日志 %(asctime)s - %(levelname)s - %(message)s')
logging.info('===== startup.py 启动成功 ======')
executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)
app_ready = False
logging.info('===== 加载模型中 ======')
client = None
def load_model():
    global client,app_ready
    client = run_vllm_server()
    app_ready = True
    logging.info('===== 模型加载完成 ======')
#client = run_vllm_server()
#logging.info('===== 模型加载完成 ======')
#app_ready = True
t=threading.Thread(target=load_model)
t.start()

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/ready", methods=["GET"]) 
def ready() -> Dict[str, bool]:  
    if app_ready:  
        return {"ready": True}  
    else:  
        abort(503)


@app.route("/api/v1/chat/completions", methods=["POST"])
def chat():
    model = request.json.get("model")
    messages = request.json.get("messages")
    temperature = request.json.get("temperature", 1.0)
    top_p = request.json.get("top_p", 1.0)
    n = request.json.get("n", 1)
    stream = request.json.get("stream", False)
    max_tokens = request.json.get("max_tokens", 2048)

    input_content = messages[0].get("content") if messages and len(messages) > 0 else ""

    if input_content.startswith("<|CHOICE|>"):
        input_content = input_content[10:]
        logging.info("Single-choice question")
        format_content = Prompt.choice_prompt_format_v1
        new_content = format_content.format(
            language=args.language,
            question=input_content
        ) if input_content else ""
    else:
        logging.info("Writing question")
        format_content = Prompt.writing_prompt_format_v1
        new_content = format_content.format(
            language=args.language,
            question=input_content
        ) if input_content else ""

    new_messages = [{
        "role": "user",
        "content": new_content
    }]

    kwargs = {
        "model": "llm",
        "messages": new_messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "max_tokens": max_tokens
    }
    result = executor.submit(process_data,client,kwargs)
    response = result.result()
    if stream:
        # logging.info("Streaming Mode")
        # True streaming response
        def generate():
            #response = client.chat.completions.create(**kwargs)
            for chunk in response:
                yield chunk.model_dump_json()
        return Response(stream_with_context(generate()), content_type="application/json")
    else:
        # logging.info("Standard Mode")
        # Standard response
        #response = client.chat.completions.create(**kwargs).model_dump_json()
        response = json.loads(response.model_dump_json())
        if not 'error' in response:
            return jsonify(response)
        else:
            return jsonify({"error": "Unexpected error occurred", "details": response.text}), response.status_code
def process_data(client,kwargs):
    response = client.chat.completions.create(**kwargs)
    return response

if __name__ == "__main__":
    app.run("0.0.0.0", 80)