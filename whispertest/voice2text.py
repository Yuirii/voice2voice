import whisper
import time

timestampe1 = time.clock_gettime(time.CLOCK_MONOTONIC)

# 加载模型 (默认为 "small")
model = whisper.load_model("small")

timestampe2 = time.clock_gettime(time.CLOCK_MONOTONIC)
print("加载模型耗时：", timestampe2-timestampe1)

# 音频文件路径
audio_file = "airpods_recording.wav" # 替换成你的音频文件路径
# 进行转录
result = model.transcribe(audio_file)
# 打印转录文本
print(result["text"])

timestampe4 = time.clock_gettime(time.CLOCK_MONOTONIC)
print("转录耗时：", timestampe4-timestampe2)

