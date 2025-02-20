import asyncio
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures

async def get_audio_data(queue): # 将 queue 作为参数传入
    print("get_audio_data start")
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    paudio = pyaudio.PyAudio()
    stream = paudio.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         frames_per_buffer=CHUNK,
                         input=True)

    print("Try Recording...")
    try:
        loop = asyncio.get_running_loop() # 获取事件循环
        while True:
            data = await loop.run_in_executor( # 使用 run_in_executor
                None, stream.read, CHUNK # 将 stream.read 放到线程池执行
            )
            waveform = np.frombuffer(data, dtype=np.int16)
            await queue.put(waveform) # 将数据放入队列
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        print("get_audio_data cancelled")
    except Exception as e: # 捕获其他可能出现的异常
        print(f"Error in get_audio_data: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        paudio.terminate()
        print("get_audio_data finished and resources cleaned up")


async def plot_audio_data(queue):
    print("plot_audio_data start")
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-')
    ax.set_xlim(0, 4096) # 使用 CHUNK 变量
    ax.set_ylim(-30000, 30000) # 调整 y 轴范围以适应 int16 数据

    try:
        while True:
            waveform = await queue.get()
            print("Try Plotting...")
            if len(waveform) > 0:
                line.set_xdata(np.arange(len(waveform)))
                line.set_ydata(waveform)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw() # 使用 fig.canvas.draw() 而不是 plt.draw()
                fig.canvas.flush_events() # 确保 GUI 事件得到处理
                # plt.pause(0.001) #  flush_events 已经包含了 pause 的功能，不需要额外的 pause
    except asyncio.CancelledError:
        print("plot_audio_data cancelled")
    except Exception as e: # 捕获其他可能出现的异常
        print(f"Error in plot_audio_data: {e}")
    finally:
        print("plot_audio_data finished")
        plt.ioff() # 关闭交互模式
        plt.show(block=False) #  非阻塞显示图像，允许程序继续运行


async def main():
    print("main start")
    queue = asyncio.Queue()
    # print("queue created")
    producer = asyncio.create_task(get_audio_data(queue)) # 将 queue 传递给 producer
    # print("producer created")
    consumer = asyncio.create_task(plot_audio_data(queue))
    # print("consumer created")

    print("Try Running...")
    try:
        await asyncio.gather(producer, consumer) # 使用 asyncio.gather 并发运行 producer 和 consumer
    except KeyboardInterrupt: # 捕获 KeyboardInterrupt 异常，方便 Ctrl+C 停止
        print("KeyboardInterrupt caught, cancelling tasks...")
        producer.cancel()
        consumer.cancel()
        await asyncio.gather(producer, consumer, return_exceptions=True) # 等待任务结束，即使被取消
    except Exception as e: # 捕获其他可能出现的异常
        print(f"Error in main: {e}")
    finally:
        print("main finished")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e: # 捕获 RuntimeError 异常，例如 "cannot reuse already awaited coroutine"
        print(f"RuntimeError in asyncio.run: {e}")
    except Exception as e: # 捕获其他顶层异常
        print(f"Top-level exception: {e}")
