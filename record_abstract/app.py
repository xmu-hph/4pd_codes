import numpy as np
import threading
import subprocess
from openai import OpenAI, APIConnectionError
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
import spacy
import json
from vllm_utils import run_vllm_server

class Myargs(BaseModel):
    device: str = 'cuda'
args = Myargs()
app = FastAPI()

logger.info(f"预加载大语言模型和分段模型")
big_model = None
tokenizer_model_zh = None
flag = False
device = args.device if torch.cuda.is_available() else "cpu"
logger.info(f"模型存放位置为:{device}")
tokenizer_path_zh = '/root/model/zh/zh_core_web_sm-3.7.0/zh_core_web_sm/zh_core_web_sm-3.7.0'
def load_model():
    logger.info(f"加载模型阶段")
    global big_model, tokenizer_model_zh, flag, device, tokenizer_path_zh
    big_model = run_vllm_server()
    torch.cuda.empty_cache()
    logger.info(f"大语言模型预热完成")
    tokenizer_model_zh = spacy.load(tokenizer_path_zh)
    logger.info(f"分段模型加载完成")
    flag = True
    logger.info("模型加载过程结束")

t = threading.Thread(target=load_model)
t.start()

@app.get("/ready")
async def check_ready():
    if not flag:
        return JSONResponse(status_code=400, content={"status_code": 400})
    return JSONResponse(status_code=200, content={"status_code": 200})

class VoiceCloneRequest(BaseModel):
    content: str

@app.post("/api/v1/generation/summarization")
async def internal_voice_clone(request: VoiceCloneRequest):
    global big_model, tokenizer_model_zh, flag, device, tokenizer_path_zh
    try:
        transcription = request.content
        texts = transcription
        prompt = f"下面是一个中文会议纪录，有多个参会人员，请你按会议主要语言抽取要句。可以理解为：输入是切片后的会议逐字稿，输出是按原句顺序排序拼接为一个段落的要句原句。会议记录内容为：{texts}"
        response = big_model.chat.completions.create(
              model="llm",
              messages=[
                {"role": "user", "content": prompt}
              ]
            )
        response = json.loads(response.model_dump_json())
        all_res = response['choices'][0]['message']['content'].strip('"')
        torch.cuda.empty_cache()
        '''
        texts = tokenizer_model_zh(transcription)
        all_res = ''
        for par in texts.sents:
            #logger.info(par.text)
            logger.info(f"本段数据长度为：{len(par.text)}")
            prompt = f"下面的内容可能是段落或非段落，需要你从给出的内容中尽可能提取出摘要句子，注意直接输出摘要句子，不用包含其他内容。如果没有摘要句子或者不是段落则输出句号'。'，给定的内容为：{par.text}"
            response = big_model.chat.completions.create(
              model="llm",
              messages=[
                {"role": "user", "content": prompt}
              ]
            )
            #logger.info(f"temp response:{response}")
            response = json.loads(response.model_dump_json())
            logger.info(f"json load response:{response}")
            all_res += response['choices'][0]['message']['content'].strip('"')
            #all_res.append(response['choices'][0]['message']['content'])
            torch.cuda.empty_cache()
        '''
        
        return JSONResponse(status_code=200, content={"summarization":{"paragraphSummary":all_res}})
    except:
        return JSONResponse(status_code=200, content={"summarization":{"paragraphSummary":"error"}})