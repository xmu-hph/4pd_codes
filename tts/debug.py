import asyncio
import websockets
import json

async def send_tts_request():
    uri = "ws://172.28.4.43:80/stream/tts"
    async with websockets.connect(uri) as websocket:
        # 配置消息
        config_message = {
            "language": "en",
            "voice_name": "default",
            "sample_rate": 44100,
            "channel": 1,
            "format": "wav",
            "bits": 16
        }
        await websocket.send(json.dumps(config_message))
        response = await websocket.recv()
        print(response)

        # 发送TTS请求
        tts_message = {
            "text": "Hello, this is a test message."
        }
        await websocket.send(json.dumps(tts_message))
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if data.get("audio_status") == 2:
                break
            print(data)

asyncio.get_event_loop().run_until_complete(send_tts_request())
