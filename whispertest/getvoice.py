import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np

p = pyaudio.PyAudio()
print("device is available: ")
for i in range(p.get_device_count()):
    device  = p.get_device_info_by_index(i)
    print(f"Index: {i}: {device['name']}, inputchannels: {device['maxInputChannels']}, outputchannels: {device['maxOutputChannels']}")

# -----  配置参数  -----
CHUNK = 1024  # 每次读取的数据块大小
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 声道数 (单声道)
RATE = 16000  # 采样率 (Hz)
RECORD_SECONDS = 5  # 录音时长 (秒)
WAVE_OUTPUT_FILENAME = "airpods_recording.wav" # 保存的文件名

# -----  AirPods 设备索引 (替换为你的AirPods索引号!) -----
AIRPODS_DEVICE_INDEX = 1  #  <----  请替换为你步骤一中找到的 AirPods 索引号

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=AIRPODS_DEVICE_INDEX, # 指定输入设备索引
                )

print(f"* 录音开始，使用设备索引 {AIRPODS_DEVICE_INDEX} (AirPods)...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

plt.plot(np.frombuffer(data, dtype=np.int16))
plt.show()

print("* 录音结束")

stream.stop_stream()
stream.close()
p.terminate()

# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()
# print(f"录音已保存到文件: {WAVE_OUTPUT_FILENAME}")