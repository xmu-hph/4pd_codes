import asyncio
import websockets
import json
import base64
import numpy as np
import torch
import time
import torchaudio
import nest_asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

nest_asyncio.apply()

debug = FastAPI()

class TTSRequest(BaseModel):
    uri: str
    text: str

async def receive_audio_data(uri, text):
    async with websockets.connect(uri) as websocket:
        rate = 24000
        config_message = json.dumps({
            "language": "zh",
            "voice_name": "default",
            "sample_rate": rate,
            "channel": 1,
            "format": "pcm",
            "bits": 16
        })
        await websocket.send(config_message)

        response = await websocket.recv()
        print("Config response:", response)
        start = time.time()
        text_message = json.dumps({"text": text})
        await websocket.send(text_message)

        audio_data_list = []
        flag = True
        while flag:
            try:
                message = await websocket.recv()
                response = json.loads(message)
                seq = response.get("audio_block_seq")
                audio_status = response.get("audio_status")
                base64_audio_data = response.get("data")
                print(seq, audio_status)
                audio_data = base64.b64decode(base64_audio_data)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_data_list.append(torch.tensor(audio_array))
                print(time.time() - start)
                start = time.time()
                if audio_status == 2:
                    print("here break")
                    flag = False
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
        result = torch.cat(audio_data_list)
        output_file = f'output_stream_{rate}.wav'
        torchaudio.save(output_file, result.unsqueeze(0), rate)
        print(f"音频文件已保存为{output_file}")
        return output_file

def run_receive_audio_data(uri, text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(receive_audio_data(uri, text))

@debug.post("/tts")
async def tts(request: TTSRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_receive_audio_data, request.uri, request.text)
    return {"message": "Request received, processing in background."}
