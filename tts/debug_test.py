import pytest
from fastapi.testclient import TestClient
import app  # 假设FastAPI应用保存在main.py

client = TestClient(app)

@pytest.fixture
def example_request():
    return {
        "text": "这是一个测试文本。",
        "voice_name": "zh-c-1",
        "language": "zh"
    }
import base64
import numpy as np
import torchaudio
import torch
def test_synthesize_speech(example_request):
    response = client.post("/v1/tts", json=example_request)
    assert response.status_code == 200
    assert "audio" in response.json()
    data = response.json()['audio']
    data = np.frombuffer(base64.b64decode(data), dtype=np.float32)
    torchaudio.save('output.wav', torch.tensor(data).unsqueeze(0), 24000)
    
