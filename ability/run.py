from flask import Flask, request, Response, jsonify, stream_with_context, abort
import requests
import os
import json
import logging
from typing import Dict
import re
from prompt import QwenPrompt as Prompt
import traceback
import time

from scripts import *


app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='# 日志 %(asctime)s - %(levelname)s - %(message)s')

LANG = os.environ.get("lang", "俄语")
USE_BAGGING = int(os.environ.get("use_bagging", False))
USE_SCORE = int(os.environ.get("use_score", False))
USE_TOC = int(os.environ.get("use_toc", False))
USE_REMOTE = int(os.environ.get("use_remote", False))
ADD_ROLE = int(os.environ.get("add_role", False))

logging.info(f'LANG is {LANG}')
logging.info(f'USE_BAGGING is {USE_BAGGING}')
logging.info(f'USE_SCORE is {USE_SCORE}')
logging.info(f'USE_TOC is {USE_TOC}')
logging.info(f'USE_REMOTE is {USE_REMOTE}')
logging.info(f'ADD_ROLE is {ADD_ROLE}')

logging.info('===== startup.py 启动成功 ======')

app_ready = False
logging.info('===== 加载模型中 ======')

if USE_REMOTE == 1:
    client, model_id = creat_client()
elif USE_REMOTE == -1:
    from vllm_utils import run_vllm_server
    client = run_vllm_server(skip=True)
    model_id = "llm"
else:
    from vllm_utils import run_vllm_server
    client = run_vllm_server()
    model_id = "llm"

logging.info('===== 模型加载完成 ======')
app_ready = True


def call_open_ai_with_retry(client, **kwargs):
    retry_times = 5
    while retry_times > 0:
        retry_times -= 1
        try:
            return client.chat.completions.create(**kwargs)
        except:
            time.sleep(1)
            traceback.print_exc()
    return None
            


# 模板信息
# for key, val in Prompt.__dict__.items():
#     if not key.startswith("_"):
#         logging.info(f"Prompt -> {key}: {val}")


@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/ready", methods=["GET"]) 
def ready() -> Dict[str, bool]:  
    if app_ready:  
        return {"ready": True}  
    else:  
        abort(503)

def get_prompt(input_content):
    """
    生成 prompt
    """
    is_choice = False
    is_test = False
    prompt_list = []
    if input_content.startswith("<|CHOICE|>"):
        logging.info(f"Single-choice question, Use toc [{USE_TOC}]")
        input_content = input_content[10:]
        if ADD_ROLE:
            # 使用角色
            for role, prompt in Prompt.choice_prompt_format_v4:
                prompt_list.append({"role": role, "content": prompt.format(language=LANG, question=input_content)})
        else:
            if USE_TOC:
                # 思维链
                prompt_list.append(
                    Prompt.choice_prompt_format_v2.format(
                    language=LANG,
                    question=input_content)
                )
                prompt_list.append(
                    Prompt.choice_prompt_format_summery_v2
                )
            else:
                # 直接推理
                prompt = Prompt.choice_prompt_format_v1
                # prompt = Prompt.choice_prompt_format_v3
                prompt_list.append(
                    prompt.format(
                    language=LANG,
                    question=input_content)
                )
            prompt_list = [{"role": "user", "content": prompt} for prompt in prompt_list]
        is_choice = True
    elif input_content.startswith("<|TEST|>"):
        logging.info("Test")
        prompt_list.append(input_content[8:])
        is_test = True
        prompt_list = [{"role": "user", "content": prompt} for prompt in prompt_list]
    else:
        logging.info("Writing question")
        if ADD_ROLE:
            # 使用角色
             for role, prompt in Prompt.writing_prompt_format_v2:
                prompt_list.append({"role": role, "content": prompt.format(language=LANG, question=input_content)})
        else:
            format_content = Prompt.writing_prompt_format_v1
            prompt_list.append(format_content.format(
                language=LANG,
                question=input_content)
            )
            prompt_list = [{"role": "user", "content": prompt} for prompt in prompt_list]
    
    return prompt_list, is_choice, is_test


def history_call(prompt_list, history, **kwargs):
    answer = ""
    rsp = None
    for prompt in prompt_list:
        history.append(prompt)
        if len(history) > 10:
            history = history[-10:]
        kwargs["messages"] = history
        rsp = call_open_ai_with_retry(client, **kwargs)
        # rsp = client.chat.completions.create(**kwargs)
        answer = rsp.choices[0].message.content
        # 记录回答
        history.append({"role": "assistant", "content": answer})
    # logging.info(history)
    return answer, rsp 


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
    prompt_list, is_choice, is_test = get_prompt(input_content)
    kwargs = {
        "model": model_id,
        # "messages": new_messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "max_tokens": max_tokens
    }


    if stream:
        logging.info("Streaming Mode")
        # True streaming response
        def generate():
            response = call_open_ai_with_retry(client, messages=prompt_list, **kwargs)
            # response = client.chat.completions.create(messages=prompt_list, **kwargs)
            # answer, response = history_call(prompt_list, history=[], **kwargs)
            for chunk in response:
                yield chunk.model_dump_json()
        return Response(stream_with_context(generate()), content_type="application/json")
    else:
        logging.info("Standard Mode")
        # Standard response
        if is_test:
            answer, response = history_call(prompt_list, history=[], **kwargs)
            response = response.model_dump_json()
            # response = client.chat.completions.create(**kwargs).model_dump_json()
            response = json.loads(response)
            
        elif (is_choice and not USE_BAGGING) or (not is_choice and not USE_SCORE):
            # response = client.chat.completions.create(**kwargs).model_dump_json()

            if ADD_ROLE:
                answer, response = history_call(prompt_list[-1:], history=prompt_list[:-1], **kwargs)
            else:  
                answer, response = history_call(prompt_list, history=[], **kwargs)
            response = response.model_dump_json()
            response = json.loads(response)
        else:
            if USE_SCORE:
                logging.info("USE_SCORE and not is_choice")
                kwargs["temperature"] = 0.7
            repeat_times = 5
            response_list = []
            ans_list = []
            for _ in range(repeat_times):   
                # response = client.chat.completions.create(**kwargs)

                answer, response = history_call(prompt_list, history=[], **kwargs)
                response = response.model_dump_json()
  
                # content = response.choices[0].message.content.strip()
                # response = response.model_dump_json()
                response = json.loads(response)
               
                if not 'error' in response:
                    response_list.append(response)
                    ans_list.append(answer)


                
            if is_choice:
                #  bagging，投票
                logging.info(f"choice_list  is:\n{ans_list}")
                # ans_list = [item.strip()[0] for item in ans_list]
                most_common_element = most_frequent(ans_list)
                if most_common_element is None:
                    pass
                else:
                    first_index = ans_list.index(most_common_element)  
                    response = response_list[first_index]
            else:
                #  加入评分机制
                score_list = []
                for content in ans_list:
                    new_content = Prompt.score_format_v1.format(language=LANG, question=input_content, answer=content)
                    new_messages = [{
                        "role": "user",
                        "content": new_content
                    }]
                    logging.info(f"score str is:\n{new_content}")
                    kwargs["messages"] = new_messages
                    response = call_open_ai_with_retry(client, **kwargs)
                    # response = client.chat.completions.create(**kwargs)
                    score_list.append(response.choices[0].message.content.strip())
                logging.info(f"score_list  is:\n{score_list}")
                score_list = [get_score(item) for item in score_list]
                first_index = score_list.index(max(score_list))
                response = response_list[first_index]
                
                
                    
            # else:
            #     # 找到最长字符串的索引
            #     max_index = max(range(len(content_list)), key=lambda index: len(content_list[index]))
            #     response = response_list[max_index]

        if not 'error' in response:
            return jsonify(response)
        else:
            return jsonify({"error": "Unexpected error occurred", "details": response.text}), response.status_code


if __name__ == "__main__":
    app.run("0.0.0.0", 80, debug=True)
