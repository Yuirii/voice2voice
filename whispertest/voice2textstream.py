import pyaudio
import whisper
import wave
import numpy as np
from io import BytesIO
import threading
import queue
import time
import noisereduce as nr
from webrtcvad import Vad

class AudioTranscriber:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name)
        self.audio_queue = queue.Queue()
        self.recording = False
        self.vad = Vad(1)  # 使用中等敏感度
        self.context = np.array([], dtype=np.float32)
        self.sample_rate = 16000

    def get_audio_devices(self):
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            device = p.get_device_info_by_index(i)
            if device['maxInputChannels'] > 0:
                devices.append((i, device['name']))
        p.terminate()
        return devices

    def save_to_wav_buffer(self, frames):
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        buffer.seek(0)
        return buffer

    def record_audio(self, device_index):
        p = pyaudio.PyAudio()
        chunk = 480  # 30ms 帧
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=chunk,
            input_device_index=device_index
        )
        self.recording = True
        active_frames = []
        silent_frames = 0

        try:
            while self.recording:
                data = stream.read(chunk, exception_on_overflow=False)
                if len(data) != chunk * 2:
                    continue
                
                if self.vad.is_speech(data, self.sample_rate):
                    active_frames.append(data)
                    silent_frames = 0
                else:
                    silent_frames += 1
                    if silent_frames > 20 and active_frames:
                        self.audio_queue.put(active_frames.copy())
                        active_frames.clear()
        finally:
            # 处理残留音频
            if active_frames:
                self.audio_queue.put(active_frames)
            stream.stop_stream()
            stream.close()
            p.terminate()

    def transcribe_audio(self):
        print("开始转写...")
        while self.recording or not self.audio_queue.empty():
            if not self.audio_queue.empty():
                frames = self.audio_queue.get()
                wav_buffer = self.save_to_wav_buffer(frames)
                
                # 转换为numpy数组
                audio_data = np.frombuffer(wav_buffer.read(), dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0

                # 降噪处理
                try:
                    audio_clean = nr.reduce_noise(
                        y=audio_data,
                        sr=self.sample_rate,
                        stationary=True
                    )
                except Exception as e:
                    print(f"降噪失败: {e}")
                    audio_clean = audio_data

                # 拼接上下文
                full_audio = np.concatenate([self.context, audio_clean])
                self.context = audio_clean[-self.sample_rate:]  # 保留1秒上下文

                # 语音识别
                result = self.model.transcribe(
                    full_audio,
                    language="zh",
                    temperature=0.1,
                    beam_size=5
                )

                # 中文去重（处理连续重复字符）
                text = result["text"].strip()
                if text:
                    dedup_text = []
                    prev_char = None
                    for char in text:
                        if char != prev_char:
                            dedup_text.append(char)
                        prev_char = char
                    final_text = ''.join(dedup_text)
                    print("识别结果:", final_text)

    def start(self, device_index):
        self.recording = True
        record_thread = threading.Thread(
            target=self.record_audio,
            args=(device_index,)
        )
        transcribe_thread = threading.Thread(target=self.transcribe_audio)
        
        record_thread.start()
        time.sleep(0.5)
        transcribe_thread.start()
        
        try:
            while self.recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.recording = False
            print("\n正在停止...")
        
        record_thread.join()
        transcribe_thread.join()

if __name__ == "__main__":
    transcriber = AudioTranscriber(model_name="small")
    devices = transcriber.get_audio_devices()
    print("可用输入设备：")
    for idx, name in devices:
        print(f"索引: {idx}, 名称: {name}")
    device_index = int(input("请输入设备索引: "))
    transcriber.start(device_index)