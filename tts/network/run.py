import torch
import torch.nn as nn
import time
class SimpleNN(nn.Module):
    def __init__(self, nums=100, max_freq=12000):
        super(SimpleNN, self).__init__()
        self.nums = nums
        self.max_freq = max_freq
        self.an = nn.Parameter(torch.randn(nums, 1))
        self.bn = nn.Parameter(torch.randn(nums, 1))
        self.a0 = nn.Parameter(torch.randn(1))
        
        # omega and alpha are now fixed but defined based on the initialized parameters
        self.register_buffer('omega', max_freq / torch.arange(1, nums + 1, dtype=torch.float).reshape(1, -1))
        self.register_buffer('alpha', self.omega / 10000)
        
        self.start_time = nn.Parameter(torch.randn(1))
    
    def forward(self, t):
        t_shifted = t - self.start_time
        exp_term = torch.exp(-t_shifted @ self.alpha)
        cos_term = torch.cos(t @ self.omega)
        sin_term = torch.sin(t @ self.omega)
        result = torch.sign(t_shifted) * (self.a0 + (exp_term * cos_term) @ self.an + (exp_term * sin_term) @ self.bn)
        
        return result

class WordVoice(nn.Module):
    def __init__(self, word=13, nums=100, max_freq=12000):
        super(WordVoice, self).__init__()
        self.word = word
        self.net = nn.ModuleList([SimpleNN(nums, max_freq) for _ in range(word)])
    
    def forward(self, t):
        out = torch.stack([net(t) for net in self.net]).sum(dim=0)
        return out
from loguru import logger
#使用sf读取wav文件
import soundfile as sf
# 指定要读取的wav文件路径
file_path = '../examples/东西.wav'
# 使用soundfile库读取wav文件
data, samplerate = sf.read(file_path)
logger.info(f'音频数据 shape: {data.shape}')
logger.info(f'采样率: {samplerate} Hz')

import matplotlib.pyplot as plt
import numpy as np
# 绘制频谱图
freq = samplerate
T = 1 / freq  # 采样周期
L = len(data)/freq  # 信号长度
N = len(data)  # 采样点数量
t = np.array(list(range(len(data))))*T
x_t = data
# 绘制时间域信号 x_discrete(t)
plt.figure(figsize=(10, 4))
plt.stem(t[11000:12000], x_t[11000:12000], basefmt=" ")
plt.title('$x_{discrete}(t)$')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig(f'Amplitude_Time.png', dpi=300, bbox_inches='tight')
#plt.show()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
if __name__=='__main__':
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WordVoice().to(device)
    x = t.reshape(-1,1).astype(np.float32)
    y = x_t.reshape(-1,1).astype(np.float32)
    x_tensor = torch.from_numpy(x).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练模型
    num_epochs = 20000
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 400 == 0:
            plt.figure(figsize=(10, 4))
            plt.plot(x[11000:12000], y[11000:12000], label='True')
            plt.plot(x[11000:12000], outputs.detach().cpu().numpy()[11000:12000], label='Predicted')
            plt.legend()
            plt.savefig(f'Predict_Amplitude_Time_{epoch+1}.png', dpi=300, bbox_inches='tight')
            #plt.show()
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {time.time()-start_time}s')
        start_time = time.time()
    # 预测
    model.eval()
    predicted = model(x_tensor).detach().cpu().numpy()
    # 绘制结果
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label='True')
    plt.plot(x, predicted, label='Predicted')
    plt.legend()
    plt.savefig(f'Predict_Amplitude_Time.png', dpi=300, bbox_inches='tight')
    #plt.show()
    #for key in model.parameters():
    #    print(key)