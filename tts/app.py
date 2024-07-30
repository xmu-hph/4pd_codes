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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_workers',type=int,default=1)
parser.add_argument('--split_length',type=int,default=80)
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--temperature',type=float,default=0.7)
parser.add_argument('--sample_rate',type=int,default=24000)
args = parser.parse_args()

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI()
# Initial arguments

# Language map
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

# Global variables
flag = False
device = args.device if torch.cuda.is_available() else "cpu"
logger.info(device)
model = None
config = None
single_length = args.split_length
sample_rate = args.sample_rate
temperature = args.temperature

# Load model function
def load_model():
    global model, device, flag, config
    config = XttsConfig()
    config.load_json("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/", eval=True)
    model = model.to(device)
    model.synthesize(
        "It took me quite a long time to develop a voice.",
        config,
        speaker_wav="./examples/default.wav",
        language="en",
    )
    torch.cuda.empty_cache()
    flag = True
    logger.info("model load complete")

# Start loading the model in a separate thread
t = threading.Thread(target=load_model)
t.start()

# Endpoint to check if the model is ready
@app.get("/ready")
async def check_ready():
    if not flag:
        return JSONResponse(status_code=400, content={"status_code": 400})
    return JSONResponse(status_code=200, content={"status_code": 200})

# Request model for voice cloning
class VoiceCloneRequest(BaseModel):
    voice_name: str
    audios: list

@app.post("/internal/v1/voice-clone")
async def internal_voice_clone(request: VoiceCloneRequest):
    try:
        audio_bytes = request.audios[0]['audio_bytes']
        audio_format = request.audios[0]['audio_format']
        voice_name = request.voice_name

        audio_data = np.frombuffer(base64.b64decode(audio_bytes), dtype=np.int16)
        #if max(abs(audio_data)) > 1:
        #    audio_data = audio_data.astype(np.int16)
        #else:
        #    audio_data = audio_data.astype(np.float32)

        torchaudio.save(f'./examples/{voice_name}.wav', torch.tensor(audio_data).unsqueeze(0), 16000)
        return JSONResponse(status_code=200, content={"voice_name": voice_name, 'status': 2})
    except:
        return JSONResponse(status_code=200, content={"voice_name": voice_name, 'status': 3})

# Request model for speech synthesis
class TTSRequest(BaseModel):
    text: str
    voice_name: str
    language: str
'''
class TTSRequest(BaseModel):
    transcription: str
    voice_name: str
    language: str
'''
#@app.post("/")
@app.post("/v1/tts")
async def synthesize_speech(request: TTSRequest):
    logger.info(f"非流式合成请求")
    global model, config, args, single_length
    transcription = request.text
    voice_name = request.voice_name
    language = request.language
    if lang_map[language] != "zh-cn":
        single_length = int(200/80*single_length)
    if voice_name == 'default':
        voice_name = voice_name
    elif voice_name[2:] in ['-c-1', '-f-standard-1', '-f-sweet-2', '-m-calm-1', '-m-standard-1']:
        voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
    else:
        voice_name = voice_name
    speaker_wav = f"./examples/{voice_name}.wav"
    logger.info(f"音色和语言请求：{voice_name, language}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
    pattern = r'[.!?;:(){}\[\]"\'。！？；：（）《》【】“”‘’、,，…]'
    text = re.split(pattern, transcription)
    post_text = []
    for par in text:
        if len(par) > single_length:
            post_text.append(par[:len(par)//2])
            post_text.append(par[len(par)//2:])
        else:
            if len(par) != 0:
                post_text.append(par)
    #logger.info(post_text)
    text = post_text
    #logger.info(text)
    logger.info(f"文本分词完成")
    chunks = []
    for index in range(len(text)):
        chunk = model.inference(text[index], lang_map[language], gpt_cond_latent, speaker_embedding, temperature=args.temperature)
        chunks.append(torch.tensor(chunk['wav']))
        torch.cuda.empty_cache()
    result = torch.cat(chunks)
    logger.info(f"数据拼接完成")
    resampled_audio = librosa.resample(result.cpu().numpy(), orig_sr=24000, target_sr=args.sample_rate)
    logger.info(f"重采样完成")
    return JSONResponse(status_code=200, content={"audio": base64.b64encode(resampled_audio.tobytes()).decode('utf-8')})

# WebSocket for streaming TTS
class StreamTTSConfig(BaseModel):
    language: str = 'en'
    voice_name: str = 'default'
    sample_rate: int = 24000
    channel: int = 1
    format: str = 'wav'
    bits: int = 16

stream_tts_config = StreamTTSConfig()

@app.websocket("/stream/tts")
async def websocket_endpoint(websocket: WebSocket):
    logger.info(f"流式合成请求")
    global single_length
    await websocket.accept()
    logger.info("流式合成连接建立")
    try:
        receive_stage = 0
        while True:
            logger.info(f"流式合成的第{receive_stage}阶段")
            message = await websocket.receive_text()
            #logger.info(message)
            data = json.loads(message)
            #logger.info(data)
            if 'language' in data:
                logger.info(f"第{receive_stage}阶段是config阶段：{data}")
                language = data["language"]
                voice_name = data["voice_name"]
                sample_rate = data["sample_rate"]
                channel = data["channel"]
                audio_format = data["format"]
                bits = data["bits"]
                stream_tts_config = StreamTTSConfig(language=language, voice_name=voice_name, sample_rate=sample_rate, channel=channel, format=audio_format, bits=bits)
                await websocket.send_text(json.dumps({"success": True}))
            elif "text" in data:
                text = data["text"]
                logger.info(f"第{receive_stage}阶段是text阶段，长度为：{len(text)}")
                i = 0
                b_start = bytes(128)
                base64_audio_data = base64.b64encode(b_start).decode('utf-8')
                start_re = {
                    "data": base64_audio_data,
                    "audio_status": 1,
                    "audio_block_seq": i
                }
                await websocket.send_text(json.dumps(start_re))
                logger.info(f"首字响应,sent {i} part")
                i += 1
                language = stream_tts_config.language
                voice_name = stream_tts_config.voice_name
                sample_rate = stream_tts_config.sample_rate
                channel = stream_tts_config.channel
                audio_format = stream_tts_config.format
                bits = stream_tts_config.bits
                logger.info(f"合成的参数：{stream_tts_config}")
                if lang_map[language] != "zh-cn":
                    single_length = int(200/80*single_length)
                if voice_name == 'default':
                    voice_name = voice_name
                elif voice_name[2:] in ['-c-1', '-f-standard-1', '-f-sweet-2', '-m-calm-1', '-m-standard-1']:
                    voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
                else:
                    voice_name = voice_name
                speaker_wav = './examples/' + voice_name + '.wav'
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
                pattern = r'[.!?;:(){}\[\]"\'。！？；：（）《》【】“”‘’、,，…]'
                text = re.split(pattern, text)
                post_text = []
                for par in text:
                    if len(par) > single_length:
                        post_text.append(par[:len(par)//2])
                        post_text.append(par[len(par)//2:])
                    else:
                        if len(par) != 0:
                            post_text.append(par)
                #logger.info(post_text)
                text = post_text
                #logger.info(text)
                logger.info(f"文本分词完成")
                for index in range(len(text)):
                    logger.info(f"text index:{index}")
                    chunks = model.inference_stream(text[index], lang_map[language], gpt_cond_latent, speaker_embedding, temperature=args.temperature)
                    for _, chunk in enumerate(chunks):
                        #logger.info(len(chunk))
                        resampled_audio = librosa.resample(chunk.cpu().numpy(), orig_sr=24000, target_sr=sample_rate)
                        response = {
                            "data": base64.b64encode(resampled_audio.tobytes()).decode('utf-8'),
                            "audio_status": 1,
                            "audio_block_seq": i
                        }
                        await websocket.send_text(json.dumps(response))
                        logger.info(f"sent {i} part")
                        #logger.info(time.time())
                        i += 1
                    torch.cuda.empty_cache()
                b = bytes(128)
                base64_audio_data = base64.b64encode(b).decode('utf-8')
                response = {
                    "data": base64_audio_data,
                    "audio_status": 2,
                    "audio_block_seq": i
                }
                await websocket.send_text(json.dumps(response))
                logger.info(f"sent {i} part")
                i += 1
            else:
                logger.info(data)
                logger.info("Received end message")
            receive_stage += 1
    except WebSocketDisconnect:
        logger.info("Connection closed: ending")

if __name__ == "__main__":
    import uvicorn
    #["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
    uvicorn.run("app:app", host="0.0.0.0", port=80)