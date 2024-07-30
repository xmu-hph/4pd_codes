#使用sf读取wav文件
import soundfile as sf
# 指定要读取的wav文件路径
file_path = './examples/default.wav'
# 使用soundfile库读取wav文件
data, samplerate = sf.read(file_path)
# data是一个numpy数组，包含了音频数据的采样值
# samplerate是采样率，即每秒的采样数
# 打印一些基本信息
print(f'音频数据 shape: {data.shape}')
print(f'采样率: {samplerate} Hz')
# 采样点数 199228，采样率44100，采样周期1/44100，采样时长为199228/44100
# 对上面的波文件进行离散傅里叶变换。
# 目前来看，要想用这个离散傅里叶变换理论，频域采样点数也需要为N，但是我们肯定不能这样做。只能固定的做。
# 加个掩码，时长选择为1000
# 每分钟160-170个字
import matplotlib.pyplot as plt
import numpy as np
# 绘制频谱图
# 示例输入信号，假设为正弦波
freq = samplerate
T = 1 / freq  # 采样周期
L = len(data)/freq  # 信号长度为1秒
N = len(data)  # 采样点数量)
t = np.array(list(range(len(data))))*T
x_t = data
# 绘制时间域信号 x_discrete(t)
plt.figure(figsize=(10, 4))
#plt.plot(t[0:30], x_t[0:30])
plt.stem(t[1100:1600], x_t[1100:1600], basefmt=" ")
plt.title('$x_{discrete}(t)$')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.savefig(f'origin_plot.png', dpi=300, bbox_inches='tight')
#plt.show()
def plot_nums(t,N,freq,T,x_t,nums,path):
    num_points = int(N/nums)
    sample_hz=np.linspace(0, freq/2, num_points)
    #print("hz:",sample_hz[0:3])
    sample_freq = 2*np.pi*sample_hz
    #print("freq:",sample_freq[0:3])
    F_omega_a = [(np.cos(ome*T*np.array(list(range(N))))*x_t).sum() for ome in sample_freq]
    F_omega_b = [(np.sin(ome*T*np.array(list(range(N))))*x_t).sum() for ome in sample_freq]
    plt.figure(figsize=(10, 4))
    plt.stem(sample_freq, F_omega_a, basefmt=" ")
    plt.title('$x_{discrete}(t)$')
    plt.xlabel('freq [rad]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(f'{path}_Aomega_plot_{nums}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    # 从上面的频率中合成音频
    #[2*value[1]*np.cos(value[0]*t)+2*value[2]*np.sin(value[0]*t) for value in list(zip(sample_freq,F_omega_a,F_omega_b))]
    pre=0
    i=0
    for value in list(zip(sample_freq,F_omega_a,F_omega_b)):
        #print(value)
        pre += (2*value[1]*np.cos(value[0]*t)+2*value[2]*np.sin(value[0]*t))*(sample_freq[1]-sample_freq[0])
        #if i >4:
        #    break
        #i += 1
    # 绘制时间域信号 x_discrete(t)
    plt.figure(figsize=(10, 4))
    #plt.plot(t[0:30], x_t[0:30])
    plt.stem(t[1100:1600], pre[1100:1600], basefmt=" ")
    plt.title('$x_{discrete}(t)$')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.savefig(f'{path}_signal_plot_{nums}.png', dpi=300, bbox_inches='tight')
    #plt.show()
import threading
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # 提交任务并收集 future 对象
    futures = [executor.submit(plot_nums,t,N,freq,T,x_t,nums,'exec') for nums in [1000,500,300,100,50,25,10,5,1]]
    # 等待所有 future 完成，并获取结果
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(f'Result: {result}')
for nums in [1000,500,300,100,50,25,10,5,1]:
    plot_nums(t,N,freq,T,x_t,nums,'sequen')