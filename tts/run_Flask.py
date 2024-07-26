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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_workers',type=int,default=1)
parser.add_argument('--split_length',type=int,default=80)
parser.add_argument('--temperature',type=float,default=0.7)
parser.add_argument('--sample_rate',type=int,default=24000)
args = parser.parse_args()
import threading
from loguru import logger
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
flag = False
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
config = None
def load_model():
    global model,device,flag,config
    config = XttsConfig()
    config.load_json("/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/", eval=True)
    model = model.to(device)
    flag=True
    logger.info("model load complete")
t=threading.Thread(target=load_model)
t.start()
from flask import Flask,Response,jsonify,request,send_from_directory
app = Flask(__name__)
@app.route("/ready", methods=["GET"])
def check_ready():
    global flag
    if not flag:
        return Response(response=jsonify({"status_code": 400}).get_data(), status=400, mimetype='application/json')
    return Response(response=jsonify({"status_code": 200}).get_data(), status=200, mimetype='application/json')
import base64
import numpy as np
import torchaudio
@app.route('/internal/v1/voice-clone', methods=['POST'])
def internal_voice_clone():
    resp = request.get_json()
    voice_name = resp['voice_name']
    audio_bytes = resp['audios']['audio_bytes']#bytes
    audio_format = resp['audios']['audio_format']#
    try:
        audio_data = np.frombuffer(base64.b64decode(audio_bytes), dtype=np.float32)
        torchaudio.save('./clones/'+voice_name+'.wav', audio_data.unsqueeze(0), args.sample_rate)
        return Response(response=jsonify({"voice_name": voice_name,'status':2}).get_data(), status=200, mimetype='application/json')
    except:
        return Response(response=jsonify({"voice_name": voice_name,'status':3}).get_data(), status=200, mimetype='application/json')
import re
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers)
@app.route('/v1/tts', methods=['POST'])
def synthesize_speech():
    global model,executor,config,args
    resp = request.get_json()
    transcription = resp['transcription']
    voice_name = resp['voice_name']
    language = resp['language']
    if voice_name=='default':
        voice_name = voice_name
    else:
        voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
    speaker_wav=f"./examples/{voice_name}.wav"
    logger.info(f"{voice_name,language}")
    result = executor.submit(long_time_run,model,config,transcription,speaker_wav,language,args)
    file_name = result.result()
    logger.info(file_name)
    file_name = file_name[2:]
    torch.cuda.empty_cache()
    return send_from_directory('./', file_name, as_attachment=True, download_name=file_name)
import random
import hashlib
import torchaudio
def long_time_run(model,config,transcription,speaker_wav,language,args):
    #将文本分成多个段，分别返回
    file_name = './'+str(hashlib.sha256(transcription.encode('utf-8')).hexdigest())+''.join(random.choices('0123456789', k=6))+'.wav'
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
    text = transcription
    single_length = args.split_length
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
    chunks = []
    for index in range(len(text)):
        chunk = model.inference(text[index],lang_map[language],gpt_cond_latent,speaker_embedding,temperature=args.temperature)
        chunks.append(torch.tensor(chunk['wav']))
    torch.cuda.empty_cache()
    result = torch.cat(chunks)
    torchaudio.save(file_name, result.unsqueeze(0), args.sample_rate)
    return file_name

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)