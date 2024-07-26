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
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--sample_rate',type=int,default=24000)
args = parser.parse_args()
import threading
from loguru import logger
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
flag = False
device = args.device if torch.cuda.is_available() else "cpu"
model = None
config = None
single_length = args.split_length
sample_rate = args.sample_rate
temperature = args.temperature
def load_model():
    global model,device,flag,config
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
        if max(abs(audio_data))>1:
            audio_data = audio_data.astype(np.int16)
        else:
            audio_data = audio_data.astype(np.float32)
        torchaudio.save('./examples/'+voice_name+'.wav', torch.tensor(audio_data).unsqueeze(0), 16000)
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
    transcription = resp['text']
    voice_name = resp['voice_name']
    language = resp['language']
    if voice_name=='default':
        voice_name = voice_name
    elif voice_name[2:] in ['-c-1','-f-standard-1','-f-sweet-2','-m-calm-1','-m-standard-1']:
        voice_name = re.sub(r'^.{2}', 'zh', voice_name, count=1)
    else:
        voice_name = voice_name 
    speaker_wav=f"./examples/{voice_name}.wav"
    logger.info(f"{voice_name,language}")
    result = executor.submit(long_time_run,model,config,transcription,speaker_wav,language,args)
    #file_name = result.result()
    #logger.info(file_name)
    #file_name = file_name[2:]
    #return send_from_directory('./', file_name, as_attachment=True, download_name=file_name)
    data = result.result()
    torch.cuda.empty_cache()
    return Response(response=jsonify(data).get_data(), status=200, mimetype='application/json')
from flask_socketio import SocketIO, send, emit
socketio = SocketIO(app,async_mode='eventlet')
@socketio.on('connect', namespace='/stream/tts')
def handle_connect():
    print("connect")
    logger.info("Connection established: waiting for configuration")

@socketio.on('disconnect', namespace='/stream/tts')
def handle_disconnect():
    print("disconnect")
    logger.info("Connection closed: ending")
import json
stream_tts_config = {"language":None,
    "voice_name":None,
    "sample_rate":None,
    "channel":None,
    "format":None,
    "bits":None}
@socketio.on('configure', namespace='/stream/tts')
def handle_configure(message):
    global model, config,stream_tts_config
    data = json.loads(message)
    logger.info(f"Configuration received: {data}")
    language = data["language"]
    voice_name = data["voice_name"]
    sample_rate = data["sample_rate"]
    channel = data["channel"]
    audio_format = data["format"]
    bits = data["bits"]
    stream_tts_config["language"] = data["language"]
    stream_tts_config["voice_name"] = data["voice_name"]
    stream_tts_config["sample_rate"] = data["sample_rate"]
    stream_tts_config["channel"] = data["channel"]
    stream_tts_config["format"] = data["format"]
    stream_tts_config["bits"] = data["bits"]
    logger.info(f"{language, voice_name, sample_rate, channel, audio_format, bits}")
    emit('response', json.dumps({"success": True}))
    
@socketio.on('message', namespace='/stream/tts')
def handle_message(message):
    global model, single_length, lang_map,stream_tts_config,args
    data = json.loads(message)
    logger.info(f"data: {data.keys()}")
    if 'text' in data:
        i = 0
        text = data["text"]
        logger.info(f"received text: {len(text), text[0:15]}")
        b_start = bytes(10)
        base64_audio_data = base64.b64encode(b_start).decode('utf-8')
        start_re = {
            "data": base64_audio_data,
            "audio_status": 1,
            "audio_block_seq": i
        }
        emit('response', json.dumps(start_re))
        logger.info(f"sent {i} part")
        i += 1
        language = stream_tts_config["language"]
        voice_name = stream_tts_config["voice_name"]
        sample_rate = stream_tts_config["sample_rate"]
        channel = stream_tts_config["channel"]
        audio_format = stream_tts_config["format"]
        bits = stream_tts_config["bits"]
        if voice_name=='default':
            voice_name = voice_name
        elif voice_name[2:] in ['-c-1','-f-standard-1','-f-sweet-2','-m-calm-1','-m-standard-1']:
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
                emit('response', json.dumps(response))
                logger.info(f"sent {i} part")
                i += 1
            torch.cuda.empty_cache()
        b = bytes(10)
        base64_audio_data = base64.b64encode(b).decode('utf-8')
        response = {
            "data": base64_audio_data,
            "audio_status": 2,
            "audio_block_seq": i
        }
        emit('response', json.dumps(response))
        logger.info(f"sent {i} part")
        i += 1
    else:
        logger.warn(f"Unexpected message format: {data}")
        
import random
import hashlib
import torchaudio
import librosa
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
    result = torch.cat(chunks)
    resampled_audio = librosa.resample(result.cpu().numpy(), orig_sr=24000, target_sr=args.sample_rate)
    #torchaudio.save(file_name, result.unsqueeze(0), args.sample_rate)
    #return file_name
    torch.cuda.empty_cache()
    return {"audio":base64.b64encode(resampled_audio.tobytes()).decode('utf-8')}

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=80, debug=False)