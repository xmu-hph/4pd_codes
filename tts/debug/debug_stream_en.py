import asyncio
import websockets
import json
import base64
import numpy as np
import torch
import time
import torchaudio
import nest_asyncio

# 解决Jupyter中的事件循环问题
nest_asyncio.apply()

# 设置超时时间常量，单位为秒
CONNECT_TIMEOUT = 10
SEND_TIMEOUT = 5
RECEIVE_TIMEOUT = 5

async def receive_audio_data(uri):
    try:
        # 确保使用 await 来等待协程完成
        async with websockets.connect(uri) as websocket:
            # 使用 asyncio.wait_for 包装协程并设置超时
            await asyncio.wait_for(send_config(websocket), timeout=CONNECT_TIMEOUT)

            audio_data_list = []
            flag = True
            start = time.time()

            while flag:
                try:
                    # 接收数据时，添加超时时间
                    message = await asyncio.wait_for(websocket.recv(), timeout=RECEIVE_TIMEOUT)
                    response = json.loads(message)
                    seq = response.get("audio_block_seq")
                    audio_status = response.get("audio_status")
                    base64_audio_data = response.get("data")
                    print(seq, audio_status)
                    
                    # 解码Base64字符串
                    audio_data = base64.b64decode(base64_audio_data)
                    
                    # 将解码后的二进制数据转换为NumPy数组
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    audio_data_list.append(torch.tensor(audio_array))
                    
                    print("Time elapsed:", time.time() - start)
                    start = time.time()
                    
                    if audio_status == 2:
                        print("Audio stream ended.")
                        flag = False
                except asyncio.TimeoutError:
                    print("Receiving data timed out.")
                    break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server.")
                    break

            # 合并音频数据并保存为文件
            result = torch.cat(audio_data_list)
            torchaudio.save(f'output_stream_{rate}.wav', result.unsqueeze(0), rate)
            print(f"Audio file saved as output_stream_{rate}.wav")
    except asyncio.TimeoutError:
        print("Connection attempt timed out.")
    except Exception as e:
        print(f"An error occurred: {e}")

async def send_config(websocket):
    rate = 24000
    config_message = json.dumps({
        "language": "en",
        "voice_name": "zh-m-standard-1",
        "sample_rate": rate,
        "channel": 1,
        "format": "pcm",
        "bits": 16
    })
    await websocket.send(config_message)

    response = await websocket.recv()
    print("Config response:", response)

    text_message = json.dumps({
        "text": ("Once upon a time, in a faraway land, there was a small village surrounded by lush green forests "
                 "and sparkling blue rivers, where people lived in harmony with nature, tending to their farms, "
                 "raising their children, and celebrating their traditions with joy and festivity, every morning "
                 "greeted by the chirping of birds and the gentle rustling of leaves, while the evenings were filled "
                 "with the warm glow of lanterns and the sound of laughter, as the villagers gathered in the town "
                 "square to share stories, sing songs, and dance to the melodies played by the local musicians, who "
                 "were renowned for their skill and passion, creating an atmosphere of unity and happiness that "
                 "permeated every corner of the village, and although life was simple, it was rich with the love and "
                 "friendship that bound the community together, making each day a precious gift, filled with moments "
                 "of wonder, discovery, and heartfelt connections, reminding everyone that true wealth lies not in "
                 "material possessions but in the bonds we forge with one another and the beauty of the world around "
                 "us, which, if cherished and nurtured, can bring endless joy and fulfillment to our lives.")
    })
    await websocket.send(text_message)

# 在Jupyter Notebook中使用await直接运行异步函数
await receive_audio_data("ws://172.28.4.42:80/stream/tts")
