curl http://172.28.4.43:80/ready
#uvicorn app:app --host 0.0.0.0 --port 80
python <<EOF
import asyncio
import websockets
import json
import base64

async def send_tts_request():
    async with websockets.connect('ws://172.28.4.43:80/stream/tts') as websocket:
        # 构造配置消息
        config_message = {
            "language": "en",
            "voice_name": "default",
            "sample_rate": 22100,
            "channel": 1,
            "format": "PCM",
            "bits": 16
        }
        # 发送配置消息
        await websocket.send(json.dumps(config_message))
        print("Configuration message sent")

        # 等待接收服务器响应
        response = await websocket.recv()
        print(f"Server response: {response}")

        # 构造文本消息
        text_message = {
            "text": "Hello, this is a test message for TTS."
        }
        # 发送文本消息
        await websocket.send(json.dumps(text_message))
        print("Text message sent")

        # 循环接收服务器发送的音频数据
        async for message in websocket:
            data = json.loads(message)
            if data["audio_status"] == 1:
                audio_data = base64.b64decode(data["data"])
                # 处理音频数据，例如保存到文件或播放等
                print(f"Received audio block: {data['audio_block_seq']}")
                print(f"Received audio block: {data['data']}")
            elif data["audio_status"] == 2:
                print("End of audio transmission")
                break

asyncio.get_event_loop().run_until_complete(send_tts_request())
EOF