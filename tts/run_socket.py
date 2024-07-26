#从榜单到模型
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
import asyncio
import websockets
import json
import base64
import logging
import threading
import wave
import time
import random
from aiohttp import web
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_workers',type=int,default=1)
parser.add_argument('--split_length',type=int,default=80)
parser.add_argument('--ping_delay',type=int,default=1000)
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--temperature',type=float,default=0.7)
parser.add_argument('--sample_rate',type=int,default=24000)
args = parser.parse_args()
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
#加载模型
flag = False
device = None
model = None
config = None
def load_model():
    global model,device,flag,config,args
    device = args.device if torch.cuda.is_available() else "cpu"
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
    flag=True
    logging.info("model load complete")
t=threading.Thread(target=load_model)
t.start()

single_length = args.split_length
ping_delay = args.ping_delay
import re
import numpy as np
import librosa
async def tts_server(websocket, path):
    global single_length,lang_map,model,config
    if path == "/stream/tts":
        logging.info("Connection established: waiting for configuration")
        try:
            initial_message = await websocket.recv()
            data = json.loads(initial_message)
            logging.info(f"Configuration received: {data}")
            language = data["language"]
            voice_name = data["voice_name"]
            sample_rate = data["sample_rate"]
            channel = data["channel"]
            audio_format = data["format"]
            bits = data["bits"]
            logging.info(f"{language,voice_name,sample_rate,channel,audio_format,bits}")
            await websocket.send(json.dumps({"success": True}))
            #speech_synthesizer = azurespeechclient.get_speech_recognizer()
            async for message in websocket:
                #if "end" in message:
                #    break
                data = json.loads(message)
                logging.info(f"data: {data.keys()}")
                if 'text' in data:
                    i = 0
                    text = data["text"]
                    logging.info(f"received text: {len(text),text[0:15]}")
                    b_start = bytes(10)
                    base64_audio_data = base64.b64encode(b_start).decode('utf-8')
                    strat_re = {
                        "data": base64_audio_data,
                        "audio_status": 1,
                        "audio_block_seq": i
                    }
                    await websocket.send(json.dumps(strat_re))
                    logging.info(f"sent {i} part")
                    i += 1
                    if voice_name=='default':
                        voice_name = voice_name
                    else:
                        voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
                    speaker_wav = './examples/' + voice_name + '.wav'
                    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
                    pattern = r'[.!?;:(){}\[\]"\'。！？；：（）《》【】“”‘’]'
                    text = re.split(pattern, text)
                    post_text = []
                    for par in text:
                        if len(par)> single_length:
                            temp = re.split(r'[,，]', par)
                            for temp_par in temp:
                                if len(temp_par)!=0:
                                    post_text.append(temp_par)
                        else:
                            if len(par)!=0:
                                post_text.append(par)
                    text = post_text
                    #text = [text[index:index+single_length] if index+single_length <len(text) else text[index:] for index in range(0, len(text), single_length)]
                    for index in range(len(text)):
                        chunks = model.inference_stream(text[index],lang_map[language],gpt_cond_latent,speaker_embedding,temperature=args.temperature)
                        for _, chunk in enumerate(chunks):
                            #resampled_audio = librosa.resample(data, orig_sr=24000, target_sr=44000)
                            resampled_audio = librosa.resample(chunk.cpu().numpy(), orig_sr=24000, target_sr=sample_rate)
                            response = {
                                    "data": base64.b64encode(resampled_audio.tobytes()).decode('utf-8'),
                                    "audio_status": 1,
                                    "audio_block_seq": i
                                    }
                            await websocket.send(json.dumps(response))
                            logging.info(f"sent {i} part")
                            i += 1
                        torch.cuda.empty_cache()
                    #i +=1
                    b = bytes(10)
                    base64_audio_data = base64.b64encode(b).decode('utf-8')
                    response = {
                        "data": base64_audio_data,
                        "audio_status": 2,
                        "audio_block_seq": i
                    }
                    await websocket.send(json.dumps(response))
                    logging.info(f"sent {i} part")
                    i += 1
                else:
                    logging.warn(f"Unexpected message format: {data}")
        finally:
            logging.info("Connection closed: ending")

async def readiness_check(request):
    global flag
    if not flag:
        return web.Response(text="Not Ready",status=400)
    return web.Response(text="OK",status=200)

async def init_app():
    app = web.Application()
    app.router.add_get('/ready', readiness_check)
    return app

async def main():
    global ping_delay
    start_server = websockets.serve(tts_server, "0.0.0.0", 80,ping_interval=ping_delay,ping_timeout=ping_delay)
    readiness_server = web.AppRunner(await init_app())
    await readiness_server.setup()
    site = web.TCPSite(readiness_server, '0.0.0.0', 8080)
    await asyncio.gather(
        start_server,
        site.start()
    )

asyncio.get_event_loop().run_until_complete(main())
asyncio.get_event_loop().run_forever()