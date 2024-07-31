import threading
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
import spacy
from transformers import AutoModelForCausalLM, AutoTokenizer

class Myargs(BaseModel):
    device: str = 'cuda'
args = Myargs()
app = FastAPI()

logger.info(f"预加载大语言模型和分段模型")
big_model = None
big_tokenizer = None
tokenizer_model_zh = None
flag = False
device = args.device if torch.cuda.is_available() else "cpu"
logger.info(f"模型存放位置为:{device}")
big_path = '/root/model/big'
tokenizer_path_zh = '/root/model/zh'
def load_model():
    logger.info(f"加载模型阶段")
    global big_model, big_tokenizer, tokenizer_model_zh, flag, device, big_path, tokenizer_path_zh
    big_model = AutoModelForCausalLM.from_pretrained(
        big_path,
        torch_dtype="auto",
        device_map="auto")
    big_tokenizer = AutoTokenizer.from_pretrained(big_path)
    logger.info(f"大语言模型模型加载完成")
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ]
    text = big_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
    model_inputs = big_tokenizer([text], return_tensors="pt").to(device)
    generated_ids = big_model.generate(
                    **model_inputs,
                    max_new_tokens=512
                    )
    generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
    response = big_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"test_response:{response}")
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
    global big_model, big_tokenizer, tokenizer_model_zh, tokenizer_model_en, flag, device, big_path, tokenizer_path_zh, tokenizer_path_en
    try:
        transcription = request.content
        texts = tokenizer_model_zh(transcription)
        all_res = []
        for par in texts.sents:
            logger.info(f"本段数据长度为：{len(par.text)}")
            prompt = f"从下列文本中提取出摘要句:{par.text}"
            messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                        ]
            text = big_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
                )
            model_inputs = big_tokenizer([text], return_tensors="pt").to(device)
            generated_ids = big_model.generate(
                                            **model_inputs,
                                            max_new_tokens=512
                                            )
            generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                            ]
            response = big_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            all_res.append(response)
            torch.cuda.empty_cache()
        return JSONResponse(status_code=200, content={"summarization":{"paragraphSummary":''.join(all_res)}})
    except:
        return JSONResponse(status_code=200, content={"summarization":{"paragraphSummary":"error"}})