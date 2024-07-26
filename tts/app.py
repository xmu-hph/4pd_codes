import asyncio
import base64
import hashlib
import random
import re
import threading

import librosa
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
import json

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

app = FastAPI()

# Initial arguments
class Args(BaseModel):
    max_workers: int = 1
    split_length: int = 80
    temperature: float = 0.7
    device: str = 'cuda'
    sample_rate: int = 24000

args = Args()

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
        gpt_cond_len=3,
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
    audios: dict

@app.post("/internal/v1/voice-clone")
async def internal_voice_clone(request: VoiceCloneRequest):
    try:
        audio_bytes = request.audios['audio_bytes']
        audio_format = request.audios['audio_format']
        voice_name = request.voice_name

        audio_data = np.frombuffer(base64.b64decode(audio_bytes), dtype=np.float32)
        if max(abs(audio_data)) > 1:
            audio_data = audio_data.astype(np.int16)
        else:
            audio_data = audio_data.astype(np.float32)

        torchaudio.save(f'./examples/{voice_name}.wav', torch.tensor(audio_data).unsqueeze(0), 16000)
        return JSONResponse(status_code=200, content={"voice_name": voice_name, 'status': 2})
    except:
        return JSONResponse(status_code=200, content={"voice_name": voice_name, 'status': 3})

# Request model for speech synthesis
class TTSRequest(BaseModel):
    text: str
    voice_name: str
    language: str
#@app.post("/")
@app.post("/v1/tts")
async def synthesize_speech(request: TTSRequest):
    global model, config, args

    transcription = request.text
    voice_name = request.voice_name
    language = request.language

    if voice_name == 'default':
        voice_name = voice_name
    elif voice_name[2:] in ['-c-1', '-f-standard-1', '-f-sweet-2', '-m-calm-1', '-m-standard-1']:
        voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
    else:
        voice_name = voice_name

    speaker_wav = f"./examples/{voice_name}.wav"
    logger.info(f"{voice_name, language}")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
    pattern = r'[.!?;:(){}\[\]"\'。！？；：（）《》【】“”‘’]'
    text = re.split(pattern, transcription)
    post_text = []

    for par in text:
        if len(par) > single_length:
            temp = re.split(r'[,，]', par)
            for temp_par in temp:
                if len(temp_par) != 0:
                    post_text.append(temp_par)
        else:
            if len(par) != 0:
                post_text.append(par)

    text = post_text

    chunks = []
    for index in range(len(text)):
        chunk = model.inference(text[index], lang_map[language], gpt_cond_latent, speaker_embedding, temperature=args.temperature)
        chunks.append(torch.tensor(chunk['wav']))

    result = torch.cat(chunks)
    resampled_audio = librosa.resample(result.cpu().numpy(), orig_sr=24000, target_sr=args.sample_rate)

    torch.cuda.empty_cache()

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
    await websocket.accept()
    logger.info("Connection established: waiting for configuration")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            if 'text' not in data:
                language = data["language"]
                voice_name = data["voice_name"]
                sample_rate = data["sample_rate"]
                channel = data["channel"]
                audio_format = data["format"]
                bits = data["bits"]
                stream_tts_config = StreamTTSConfig(language=language, voice_name=voice_name, sample_rate=sample_rate, channel=channel, format=audio_format, bits=bits)
                logger.info(f"Configuration received: {data}")
                await websocket.send_text(json.dumps({"success": True}))
            else:
                i = 0
                text = data["text"]
                logger.info(f"received text: {len(text), text[0:15]}")

                b_start = bytes(128)
                base64_audio_data = base64.b64encode(b_start).decode('utf-8')
                start_re = {
                    "data": base64_audio_data,
                    "audio_status": 1,
                    "audio_block_seq": i
                }
                await websocket.send_text(json.dumps(start_re))
                logger.info(f"sent {i} part")
                i += 1

                language = stream_tts_config.language
                voice_name = stream_tts_config.voice_name
                sample_rate = stream_tts_config.sample_rate
                channel = stream_tts_config.channel
                audio_format = stream_tts_config.format
                bits = stream_tts_config.bits

                if voice_name == 'default':
                    voice_name = voice_name
                elif voice_name[2:] in ['-c-1', '-f-standard-1', '-f-sweet-2', '-m-calm-1', '-m-standard-1']:
                    voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
                else:
                    voice_name = voice_name

                speaker_wav = './examples/' + voice_name + '.wav'
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
                pattern = r'[.!?;:(){}\[\]"\'。！？；：（）《》【】“”‘’]'
                text = re.split(pattern, text)
                post_text = []

                for par in text:
                    if len(par) > single_length:
                        temp = re.split(r'[,，]', par)
                        for temp_par in temp:
                            if len(temp_par) != 0:
                                post_text.append(temp_par)
                    else:
                        if len(par) != 0:
                            post_text.append(par)

                text = post_text
                for index in range(len(text)):
                    chunks = model.inference_stream(text[index], lang_map[language], gpt_cond_latent, speaker_embedding, temperature=args.temperature)
                    for _, chunk in enumerate(chunks):
                        resampled_audio = librosa.resample(chunk.cpu().numpy(), orig_sr=24000, target_sr=sample_rate)
                        response = {
                            "data": base64.b64encode(resampled_audio.tobytes()).decode('utf-8'),
                            "audio_status": 1,
                            "audio_block_seq": i
                        }
                        await websocket.send_text(json.dumps(response))
                        logger.info(f"sent {i} part")
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

    except WebSocketDisconnect:
        logger.info("Connection closed: ending")
