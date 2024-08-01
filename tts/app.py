import asyncio
import base64
import hashlib
import random
import re
import threading
import time
import librosa
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
import json
import os
import spacy
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class Myargs(BaseModel):
    device: str = 'cuda'
    temperature: float = 0.7
    sample_rate: int = 24000
args = Myargs()
app = FastAPI()

lang_map = {
    "zh": "zh-cn",
    "cn": "zh-cn",
    "en": "en",
    "tr": "tr",
    "es": "es",
    "fr": "fr",
    "ja": "ja",
    "de": "de",
    "ru": "ru",
    "pt": "pt",
    "pl": "pl",
    "it": "it",
    "nl": "nl",
    "sv": "sv",
    "ar": "ar",
    "hu": "hu",
    "cs": "cs",
    "no": "no",
    "fi": "fi",
    "da": "da",
    "el": "el",
    "bg": "bg",
    "ko": "ko",
    "hi": "hi",
    "he": "he"
}
logger.info(f"预加载语音合成模型和分词模型")
tts_model = None
tokenizer_model_zh = None
tts_config = None
tokenizer_model_en = None
flag = False
device = args.device if torch.cuda.is_available() else "cpu"
logger.info(f"模型存放位置为:{device}")
tts_path = '/root/model/tts'
tokenizer_path_zh = '/root/model/zh'
tokenizer_path_en = '/root/model/en'
voice_names = ['default','zh-c-1','zh-f-standard-1','zh-f-sweet-2','zh-m-calm-1','zh-m-standard-1']
voice_features = {}
def load_model():
    logger.info(f"加载模型阶段")
    global tts_model, tokenizer_model_zh, tts_config, tokenizer_model_en, flag, device, tts_path, tokenizer_path_zh, tokenizer_path_en, voice_names, voice_features
    tts_config = XttsConfig()
    tts_config.load_json(tts_path+"/config.json")
    tts_model = Xtts.init_from_config(tts_config)
    tts_model.load_checkpoint(tts_config, checkpoint_dir=tts_path, eval=True)
    tts_model = tts_model.to(device)
    logger.info(f"tts模型加载完成")
    tts_model.synthesize(
        "It took me quite a long time to develop a voice.",
        tts_config,
        speaker_wav="./examples/zh-c-1.wav",
        language="en",
    )
    #torch.cuda.empty_cache()
    logger.info(f"tts模型预热完成")
    tokenizer_model_en = spacy.load(tokenizer_path_en)
    tokenizer_model_zh = spacy.load(tokenizer_path_zh)
    logger.info(f"分词模型加载完成")
    for voice_name in voice_names:
        speaker_wav = './examples/' + voice_name + '.wav'
        gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[speaker_wav])
        voice_features[voice_name] = {"gpt_cond_latent":gpt_cond_latent,"speaker_embedding":speaker_embedding}
    logger.info(f"音色特征提取结束")
    flag = True
    logger.info("模型加载线程结束")

t = threading.Thread(target=load_model)
t.start()

@app.get("/ready")
async def check_ready():
    if not flag:
        return JSONResponse(status_code=400, content={"status_code": 400})
    return JSONResponse(status_code=200, content={"status_code": 200})

class VoiceCloneRequest(BaseModel):
    voice_name: str
    audios: list

@app.post("/internal/v1/voice-clone")
async def internal_voice_clone(request: VoiceCloneRequest):
    logger.info(f"音色克隆请求")
    global tts_model, tokenizer_model_zh, tts_config, tokenizer_model_en, flag, device, tts_path, tokenizer_path_zh, tokenizer_path_en, voice_names, voice_features
    try:
        audio_bytes = request.audios[0]['audio_bytes']
        audio_format = request.audios[0]['audio_format']
        voice_name = request.voice_name
        audio_data = np.frombuffer(base64.b64decode(audio_bytes), dtype=np.int16)
        torchaudio.save(f'./examples/{voice_name}.wav', torch.tensor(audio_data[:min(48000,len(audio_data))]).unsqueeze(0), 16000)
        logger.info(f"音色文件保存完成")
        gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[f'./examples/{voice_name}.wav'])
        logger.info(f"音色特征提取完成")
        voice_features[voice_name] = {"gpt_cond_latent":gpt_cond_latent,"speaker_embedding":speaker_embedding}
        logger.info(f"音色特征保存完成")
        return JSONResponse(status_code=200, content={"voice_name": voice_name, 'status': 2})
    except:
        return JSONResponse(status_code=200, content={"voice_name": voice_name, 'status': 3})

class TTSRequest(BaseModel):
    text: str
    voice_name: str
    language: str

@app.post("/v1/tts")
async def synthesize_speech(request: TTSRequest):
    global tts_model, tokenizer_model_zh, tts_config, tokenizer_model_en, flag, device, tts_path, tokenizer_path_zh, tokenizer_path_en, voice_names, voice_features
    logger.info(f"非流式合成请求接入")
    transcription = request.text
    voice_name = request.voice_name
    language = request.language
    logger.info(f"总文本长度为：{len(transcription)}")
    if voice_name == 'default':
        voice_name = voice_name
    elif voice_name[2:] in ['-c-1', '-f-standard-1', '-f-sweet-2', '-m-calm-1', '-m-standard-1']:
        voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
    else:
        voice_name = voice_name
    gpt_cond_latent = voice_features[voice_name]['gpt_cond_latent']
    speaker_embedding = voice_features[voice_name]['speaker_embedding']
    logger.info(f"音色特征提取完成")
    '''
    speaker_wav = f"./examples/{voice_name}.wav"
    logger.info(f"音色和语言请求：{voice_name, language}")
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[speaker_wav])
    '''
    if lang_map[language] == "zh-cn":
        text = tokenizer_model_zh(transcription)
    elif lang_map[language] == "en":
        text = tokenizer_model_en(transcription)
    else:
        logger.info(f"不支持此语言")
    logger.info(f"文本分词完成")
    chunks = []
    for par in text.sents:
        logger.info(f"本段数据长度为：{len(par.text)}")
        chunk = tts_model.inference(par.text, lang_map[language], gpt_cond_latent, speaker_embedding, temperature=args.temperature)
        chunks.append(torch.tensor(chunk['wav']))
        #torch.cuda.empty_cache()
    logger.info(f"数据合成完成")
    result = torch.cat(chunks)
    logger.info(f"数据拼接完成")
    resampled_audio = result.cpu().numpy()
    logger.info(f"重采样完成")
    return JSONResponse(status_code=200, content={"audio": base64.b64encode(resampled_audio.tobytes()).decode('utf-8')})

class StreamTTSConfig(BaseModel):
    language: str = 'en'
    voice_name: str = 'default'
    sample_rate: int = 24000
    channel: int = 1
    format: str = 'wav'
    bits: int = 16

stream_tts_config = StreamTTSConfig()
def text_chunk_generator(text, chunk_size=30):
    """
    生成一个不超过chunk_size字符的字符串生成器
    :param text: 超长文本串
    :param chunk_size: 每个生成的字符串的最大长度
    :return: 生成器，每次生成一个不超过chunk_size字符的字符串
    """
    start = 0
    while start < len(text):
        yield text[start:start + chunk_size]
        start += chunk_size

@app.websocket("/stream/tts")
async def websocket_endpoint(websocket: WebSocket):
    global tts_model, tokenizer_model_zh, tts_config, tokenizer_model_en, flag, device, tts_path, tokenizer_path_zh, tokenizer_path_en, stream_tts_config, voice_names, voice_features
    logger.info(f"流式合成请求接入")
    await websocket.accept()
    logger.info("流式合成连接建立")
    try:
        receive_stage = 0
        while True:
            logger.info(f"流式合成的第{receive_stage}阶段")
            message = await websocket.receive_text()
            data = json.loads(message)
            logger.info(f"参数接收完成，待分析")
            if 'language' in data:
                logger.info(f"第{receive_stage}阶段是参数传递阶段，内容为{data}")
                language = data["language"]
                voice_name = data["voice_name"]
                sample_rate = data["sample_rate"]
                channel = data["channel"]
                audio_format = data["format"]
                bits = data["bits"]
                stream_tts_config = StreamTTSConfig(language=language, \
                    voice_name=voice_name, \
                    sample_rate=sample_rate, \
                    channel=channel, \
                    format=audio_format, \
                    bits=bits)
                logger.info(f"第{receive_stage}阶段，数据解析完成")
                await websocket.send_text(json.dumps({"success": True}))
                logger.info(f"第{receive_stage}阶段，信息回传完成")
            elif "text" in data:
                text = data["text"]
                logger.info(f"第{receive_stage}阶段是语音合成阶段，文本长度为：{len(text)}")
                i = 0
                b_start = bytes(128)
                base64_audio_data = base64.b64encode(b_start).decode('utf-8')
                start_re = {
                    "data": base64_audio_data,
                    "audio_status": 1,
                    "audio_block_seq": i
                }
                logger.info(f"首次空响应合成完成")
                await websocket.send_text(json.dumps(start_re))
                logger.info(f"首字空响应发送完成,sent {i} part")
                i += 1
                language = stream_tts_config.language
                voice_name = stream_tts_config.voice_name
                sample_rate = stream_tts_config.sample_rate
                channel = stream_tts_config.channel
                audio_format = stream_tts_config.format
                bits = stream_tts_config.bits
                logger.info(f"合成的参数：{stream_tts_config}")
                if voice_name == 'default':
                    voice_name = voice_name
                elif voice_name[2:] in ['-c-1', '-f-standard-1', '-f-sweet-2', '-m-calm-1', '-m-standard-1']:
                    voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
                else:
                    voice_name = voice_name
                '''
                speaker_wav = './examples/' + voice_name + '.wav'
                logger.info(f"音色信息：{speaker_wav}")
                gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[speaker_wav])
                '''
                gpt_cond_latent = voice_features[voice_name]['gpt_cond_latent']
                speaker_embedding = voice_features[voice_name]['speaker_embedding']
                logger.info(f"音色特征提取完成")
                if lang_map[language] == "zh-cn":
                    text = tokenizer_model_zh(text)
                elif lang_map[language] == "en":
                    text = tokenizer_model_en(text)
                else:
                    logger.info(f"不支持此语言")
                    break
                logger.info(f"文本分词完成")
                for par in text.sents:
                    logger.info(f"本段长度：{len(par.text)}")
                    chunks = tts_model.inference_stream(par.text, lang_map[language], gpt_cond_latent, speaker_embedding, temperature=args.temperature)
                    for _, chunk in enumerate(chunks):
                        resampled_audio = librosa.resample(chunk.cpu().numpy(), orig_sr=24000, target_sr=sample_rate)
                        response = {
                            "data": base64.b64encode(resampled_audio.tobytes()).decode('utf-8'),
                            "audio_status": 1,
                            "audio_block_seq": i
                        }
                        #logger.info(f"响应合成完成")
                        await websocket.send_text(json.dumps(response))
                        logger.info(f"sent {i} part")
                        i += 1
                    #torch.cuda.empty_cache()
                logger.info(f"文本内容发送完成")
                b = bytes(128)
                base64_audio_data = base64.b64encode(b).decode('utf-8')
                response = {
                    "data": base64_audio_data,
                    "audio_status": 2,
                    "audio_block_seq": i
                }
                logger.info(f"结束响应合成完成")
                await websocket.send_text(json.dumps(response))
                logger.info(f"结束片段发送完成")
                i += 1
            else:
                logger.info(f"Received {data} message")
            receive_stage += 1
    except WebSocketDisconnect:
        logger.info("Connection closed by Client")