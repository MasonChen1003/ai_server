# 用戶端與伺服器可在單次TCP連接，實現無限輪次對話，直至主動斷開。
import subprocess
import socket, os, time, re, wave, struct
import threading
import soundfile as sf  # 添加音訊讀取庫
import edge_tts
from openai import OpenAI
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from opencc import OpenCC # 簡轉繁的功能
import json
from rapidfuzz import process
import paho.mqtt.client as mqtt  # ✅ 新增 MQTT 支援
#>pip install paho-mqtt


Temp=None
Count=None
# FunASR語音辨識，語音轉文字
class INMP441ToWAV:
    def __init__(self):
        self.SAMPLE_RATE = 16000
        self.BITS = 16
        self.CHANNELS = 1
        self.BUFFER_SIZE = 4096

    def receive_inmp441_data(self, conn):
        audio_data = b''  # 用於累積音訊資料的緩衝區
        try:
            while True:
                # 讀取包頭
                header = conn.recv(4)
                if not header:
                    break
                data_len = struct.unpack('<I', header)[0]
                # 讀取數據體
                data = b''
                while len(data) < data_len:
                    packet = conn.recv(data_len - len(data))
                    if not packet:
                        break
                    data += packet
                if data_len == 0:  # 結束標記
                    if audio_data:
                        self.save_inmp441_wav(audio_data)
                        audio_data = b''  # 清空緩衝區
                        return "recording_1.wav"
                else:
                    audio_data += data  # 累積音訊資料
                    
        except socket.timeout:
            # 捕獲超時錯誤並返回 None
            print("接收INMP441資料時出現錯誤: timed out")
            return None
        except socket.error as e:
            # 如果錯誤代碼是 10038，代表 socket 已關閉，直接返回 None
            if hasattr(e, 'errno') and e.errno == 10038:
                return None
            print(f"接收INMP441資料時出現錯誤: {e}\n")
            if audio_data:  # 如果已有收集的資料，嘗試保存
                self.save_inmp441_wav(audio_data)
                return "recording_1.wav"
            return None

    def save_inmp441_wav(self, data):
        filename = "recording_1.wav"
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(self.BITS // 8)
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(data)
            print(f"已保存錄音檔：{filename}")
        except Exception as e:
            print(f"保存錄音檔時出現錯誤: {e}")


# FunASR語音辨識，語音轉文字
class FunasrSpeechToText:
    def __init__(self):
        try:
            # 正確載入模型
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",  # 使用標準模型ID而非本地路徑
                # model="iic/paraformer-zh-streaming",  # 使用標準模型ID而非本地路徑
                #disable_update=True,
            )
        except Exception as e:
            print(f"初始化FunASR模型失敗: {e}")
            self.model = None

    def recognize_speech(self, client_socket, audio_path):
        if not self.model:
            print("FunASR模型未正確初始化")
            self.send_end_of_stream(client_socket)
            return ""
            
        try:
            # 檢查文件是否存在
            if not os.path.exists(audio_path):
                print(f"音訊檔案不存在: {audio_path}")
                self.send_end_of_stream(client_socket)
                return ""
                
            # 正確讀取音訊資料
            speech, sample_rate = sf.read(audio_path)  # 讀取為numpy陣列
            cache = {}
            # 使用音訊陣列作為輸入
            res = self.model.generate(
                input=speech,  # 傳入音訊資料而非路徑
                input_fs=sample_rate,  # 添加取樣速率參數
                cache=cache,
                language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                is_final=False,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1
            )
            out_text = rich_transcription_postprocess(res[0]["text"]).strip()
            cc = OpenCC('s2t')  # s2t 簡轉繁
            text = cc.convert(out_text)
            # **只返回多於 1 個字的結果**
            if len(text) > 1:                              
                return str(text)
            else:
                print("⚠️ 識別結果過短，忽略輸出")
                return 

        except Exception as e:
            print(f"⚠️ 語音識別錯誤：{str(e)}")
            self.send_end_of_stream(client_socket)
            return ""
            
    def send_end_of_stream(self, client_socket):
        try:
            time.sleep(0.03)  # 結束用戶端等待伺服器返回播放資料
            client_socket.sendall("END_OF_STREAM\n".encode())
        except socket.error as e:
            print(f"發送結束標記時出現錯誤: {e}")

# groq 的回復
class groqReply:
    def __init__(self):
        self.api_key = "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"
        self.base_url = "https://api.groq.com/openai/v1"
        self.role_setting = '（習慣簡短表達，不要多行，不要回車，你是一個叫小愛的溫柔女朋友，聲音好聽，只要中文，愛用網絡梗，最後拋出一個提問。）'
        self.groq_model = 'qwen-2.5-32b'

    def get_groq_response(self, client_socket, text):
        if not text or not text.strip():
            print("輸入文本為空，不發送API請求")
            self.send_end_of_stream(client_socket)
            return ""
            
        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            response = client.chat.completions.create(
                model=self.groq_model,
                messages=[{
                    'role': 'user',
                    'content': f"{text}{self.role_setting}"
                }],
                stream=True
            )
            content_list = []
            for chunk in response:
                content = chunk.choices[0].delta.content
                content_list.append(content)
            # 1. 去掉'練習', '跑步', '需要',==練習跑步需要
            processed_sentence = ''.join([element for element in content_list if element])
            # 2.去掉  ###，- **， **
            cleaned_text = re.sub(r'### |^- \*\*|\*\*', '', processed_sentence, flags=re.MULTILINE)
            # 3.移除 DeepSeek R1 中<think>...</think> 之間的內容，只保留最終的回應
            clean_text = re.sub(r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL).strip()

            return clean_text
        except Exception as e:
            print(f"⚠️ Groq API錯誤：{str(e)}")
            self.send_end_of_stream(client_socket)
            return ""
            
    def send_end_of_stream(self, client_socket):
        try:
            time.sleep(0.03)  # 結束用戶端等待伺服器返回播放資料
            client_socket.sendall("END_OF_STREAM\n".encode())
        except socket.error as e:
            print(f"發送結束標記時出現錯誤: {e}")

# EdgeTTS文字生成語音
class EdgeTTSTextToSpeech:
    def __init__(self):
        self.voice = "zh-TW-HsiaoYuNeural"
        self.rate = '+16%'
        self.volume = '+0%'
        self.pitch = '+0Hz'
        self.communicate_path = "response.mp3"

    def generate_audio(self, client_socket, text):  # EdgeTTS文字生成語音
        if not text or not text.strip():
            print("輸入文本為空，不生成TTS")
            self.send_end_of_stream(client_socket)
            return None
            
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch)
            communicate.save_sync(self.communicate_path)
            return self.communicate_path
        except Exception as e:
            print(f"⚠️ TTS生成失敗: {str(e)}")
            self.send_end_of_stream(client_socket)
            return None
            
    def send_end_of_stream(self, client_socket):
        try:
            time.sleep(0.03)  # 結束用戶端等待伺服器返回播放資料
            client_socket.sendall("END_OF_STREAM\n".encode())
        except socket.error as e:
            print(f"發送結束標記時出現錯誤: {e}")

# FFmpeg 音訊轉換器
class FFmpegToWav:
    def __init__(self, sample_rate, channels, bit_depth):
        self.sample_rate = sample_rate
        self.channels = channels
        if bit_depth in [16, 24]:
            self.bit_depth = bit_depth
        else:
            raise ValueError("bit_depth 必須是 16 或 24")

    def convert_to_wav(self, client_socket, input_file, output_file):
        if not input_file or not os.path.exists(input_file):
            print(f"輸入檔案不存在: {input_file}")
            self.send_end_of_stream(client_socket)
            return False
            
        codec = 'pcm_s16le' if self.bit_depth == 16 else 'pcm_s24le'
        try:
            subprocess.run([
                'ffmpeg',
                '-i', input_file,  # 輸入檔
                '-vn',  # 禁用視頻流
                '-acodec', codec,  # 動態設置編碼器（根據位元深）
                '-ar', str(self.sample_rate),  # 取樣速率
                '-ac', str(self.channels),  # 聲道數
                '-y',  # 覆蓋輸出檔
                output_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            print(f"轉換成功: {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"轉換失敗: {e.stderr.decode('utf-8', errors='replace')}")
            self.send_end_of_stream(client_socket)
            return False
        except FileNotFoundError:
            print("錯誤: 未找到 FFmpeg，請確保已正確安裝並添加到系統 PATH")
            self.send_end_of_stream(client_socket)
            return False
            
    def send_end_of_stream(self, client_socket):
        try:
            time.sleep(0.03)  # 結束用戶端等待伺服器返回播放資料
            client_socket.sendall("END_OF_STREAM\n".encode())
        except socket.error as e:
            print(f"發送結束標記時出現錯誤: {e}")

# MAX98357播放聲音
class MAX98357AudioPlay:
    def __init__(self):
        self.chunk = 1024  # 音訊幀數（緩衝區大小）

    def send_wav_file(self, client_socket, wav_file_path):
        '''
        if client_socket.fileno() == -1:  # ❌ 檢查 socket 是否已關閉
            print("⚠️ 嘗試發送 WAV，但 TCP 連線已關閉，跳過傳輸")
            return
        '''
        if not os.path.exists(wav_file_path):
            print(f"WAV檔案不存在: {wav_file_path}")
            self.send_end_of_stream(client_socket)
            return
        self.running = False   
        try:
            with open(wav_file_path, "rb") as audio_file:
                audio_file.seek(44)  # 跳過前44位元組的WAV檔頭資訊
                while True:
                    chunk = audio_file.read(1024)
                    if not chunk:
                        break
                    client_socket.sendall(chunk)
            time.sleep(0.1)
            self.send_end_of_stream(client_socket)
            print("回復音訊已發送")
            self.running = True
            
        except socket.error as e:
            print(f"發送WAV檔案時出現錯誤: {e}")
        except Exception as e:
            print(f"處理WAV檔案時出現錯誤: {e}")
            self.send_end_of_stream(client_socket)
            
    def send_end_of_stream(self, client_socket):
        try:
            client_socket.sendall("END_OF_STREAM\n".encode())
        except socket.error as e:
            print(f"發送結束標記時出現錯誤: {e}")

# 小智AI伺服器 主迴圈
class XiaoZhi_Ai_TCPServer:
    
    def __init__(self, host="0.0.0.0", port=8888, save_path="audio/received_audio.wav"):
        self.host = host
        self.port = port
        self.received_audio_filename = save_path
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.fstt = FunasrSpeechToText()  # FunASR 語音辨識，語音轉文字
        self.dsr = groqReply()  # groq 的回復
        self.etts = EdgeTTSTextToSpeech()  # EdgeTTS 文字生成語音
        self.mapl = MAX98357AudioPlay()  # MAX98357 播放音訊
        self.fftw = FFmpegToWav(sample_rate=8000, channels=1, bit_depth=16)  # FFmpeg 音訊轉換器
        self.inmp441tw = INMP441ToWAV()
        self.running = False
        self.clients = {}  # 存儲客戶端連接
        
        self.commands = {}  # ✅ 存在記憶體內，不寫入檔案
        self.subscribed_topics = {}  # ✅ 存儲訂閱的 MQTT 主題及最新值

        # === MQTT 相關設定 ===
        MQTT_BROKER = "mqttgo.io"  # 你的 MQTT 伺服器
        MQTT_PORT = 1883    
        
        # ✅ 設置 MQTT Client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # ✅ 啟動 MQTT 事件迴圈（非阻塞）
        self.mqtt_client.loop_start()
        
    
    def start(self):
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            
            local_ip = socket.gethostbyname(socket.gethostname())
            print("\n=== 小智AI對話機器人伺服器_V2.0 已啟動 ===")
            print(f"IP埠為：{local_ip}:{self.port}")
            print("等待用戶端的連接...")
            
            # 開始接受連接
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            # 主線程可以處理用戶輸入或其他任務
            try:
                while self.running:
                    cmd = input("可輸入 'quit' 隨時關閉伺服器\n")
                    if cmd.lower() == 'quit':
                        print("正在關閉伺服器...")
                        self.stop()
                        break
            except KeyboardInterrupt:
                self.stop()
            
        except socket.error as e:
            print(f"伺服器啟動錯誤: {e}")
            self.stop()
    
    def accept_connections(self):
        while self.running:
            try:
                # 接受客戶端連接
                client_socket, client_address = self.socket.accept()
                print(f"接收到來自 {client_address} 的持久連接")
                
                # 設置客戶端socket超時
                client_socket.settimeout(None)
                
                # 存儲客戶端信息
                self.clients[client_address] = client_socket
                
                # 在單獨的線程中處理客戶端
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
            
            except socket.error as e:
                # 如果伺服器正在關閉或發生 WinError 10038，直接跳出迴圈
                if not self.running or (hasattr(e, 'errno') and e.errno == 10038):
                    break
                print(f"接受連接時出現錯誤: {e}")
     
    def load_commands(self):
        """返回記憶體中的 commands.json"""
        return self.commands

    def get_commands(self):
        """返回記憶體中的 commands"""
        return self.commands

    def update_commands(self, new_commands):
        """更新記憶體中的 commands.json"""
        if isinstance(new_commands, dict):
            #self.commands = new_commands  # ✅ 更新記憶體內的 JSON
            self.commands.update(new_commands)  # ✅ 追加新指令，而不覆蓋舊數據
            print("🔄 `commands.json` 已更新")
        else:
            print("❌ 更新失敗，資料格式錯誤")

    def handle_client(self, client_socket, client_address):
        global Temp,Count
        try:
            data = client_socket.recv(4096).decode("utf-8")  # 讀取來自客戶端的數據
            if data:
                # ✅ 存儲 TCP Client，讓 MQTT 可以找到對應的 Client
                self.clients[client_address] = client_socket
                
                # 解析並回應
                print(f"📩 接收到來自 {client_address} 的數據: {data}")
                try:
                    received_json = json.loads(data)  # 解析 JSON
                    print(f"收到的語言對應字典: {received_json}")  # 打印確認
                    self.update_commands(received_json)  # ✅ 更新記憶體中的指令

                except json.JSONDecodeError:
                    print("❌ 無法解析來自客戶端的 JSON")
                    self.commands=self.load_commands()
            
            while self.running:
                # 接收INMP441 麥克風資料
                inmp441wav_path = self.inmp441tw.receive_inmp441_data(client_socket)
                
                if not inmp441wav_path:
                    #print("未收到有效音訊資料，等待新資料...\n")
                    #continue
                    print(f"客戶端 {client_address} 可能已斷線，關閉連線...\n")
                    break
                
                # FunASR語音辨識，語音轉文字
                fstt_text = self.fstt.recognize_speech(client_socket, inmp441wav_path)
                print("FunASR 語音辨識---：", fstt_text)
                
                command_map=self.get_commands()
                # 使用 rapidfuzz 進行模糊匹配，並處理 None
                match_result = process.extractOne(fstt_text, command_map.keys())
    
                if match_result:  # 確保匹配結果不是 None
                    best_match, score, _ = match_result
                    if score > 50:  # 設定相似度門檻（60 以上才算匹配成功）
                        command = command_map[best_match].split("/")
                        
                        if command[0]=='pub':
                            self.mqtt_client.publish(command[1],command[2])
                            print(f"已發佈MQTT主題: {command[1]} 訊息:{command[2]}")
                            self.sayword(client_socket, f"已發佈MQTT主題:{command[1]} 訊息:{command[2]}")
                        
                        elif command[0]=='sub':
                            # ✅ 避免重複訂閱
                            if command[1] in self.subscribed_topics:
                                print(f"⚠️ 主題 [{command[1]}] 已訂閱，跳過")
                            else:
                                self.mqtt_client.subscribe(command[1])

                            value=self.get_topic_value(command[1])
                            self.sayword(client_socket, f"{best_match}{value}")
                        continue
                else:
                    print(f"未找到 {fstt_text} 的對應指令")
                
               
                # groq生成回復
                if fstt_text and fstt_text.strip():
                    gdr_text = self.dsr.get_groq_response(client_socket, fstt_text)
                    if gdr_text:
                        print("groq 的回復---：", gdr_text)
                        
                        # EdgeTTS 文字生成語音
                        tts_path = self.etts.generate_audio(client_socket, gdr_text)
                        if tts_path:
                            print("EdgeTTS 音訊地址---：", tts_path)
                            
                            # FFmpeg 音訊轉換器
                            if self.fftw.convert_to_wav(client_socket, tts_path, 'output.wav'):
                                # MAX98357 播放音訊
                                self.mapl.send_wav_file(client_socket, 'output.wav')
                    else:
                        print('Groq API返回空回復')
                        self.send_end_of_stream(client_socket)
                else:
                    print('FunASR語音辨識為空，繼續講話....')
                    self.send_end_of_stream(client_socket)
                    
        except ConnectionError as e:
            print(f"連接異常: {e}")
        except socket.error as e:
            if hasattr(e, 'errno') and e.errno == 10054:  # Windows特定錯誤(連接重置)
                print(f"連接異常: [WinError 10054] 遠端主機已強制關閉一個現存的連線。")
            else:
                print(f"處理客戶端 {client_address} 時出現錯誤: {e}")
        except Exception as e:
            print(f"處理客戶端時發生未知錯誤: {e}")
            
        finally:
            # 清理客戶端連接
            self.close_client(client_address)
            
                
    def sayword(self,client_socket, myword):
        # 生成語音並播放
        tts_path = self.etts.generate_audio(client_socket, myword)
        print("EdgeTTS 聲音：", tts_path)
        # FFmpeg 音頻轉換器
        self.fftw.convert_to_wav(client_socket, tts_path, 'output.wav')
        # MAX98357 播放音頻
        self.mapl.send_wav_file(client_socket, 'output.wav')
    
    def close_client(self, client_address):
        client = self.clients.pop(client_address, None)
        if client:
            try:
                client.close()
            except Exception as e:
                pass
            print(f"連接 {client_address} 已關閉")

    def send_end_of_stream(self, client_socket):
        try:
            time.sleep(0.03)  # 結束用戶端等待伺服器返回播放資料
            client_socket.sendall("END_OF_STREAM\n".encode())
        except socket.error as e:
            print(f"發送結束標記時出現錯誤: {e}")
    
    def stop(self):
        self.running = False
        # 嘗試先關閉傳輸，通知所有等待中的操作停止
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except Exception as e:
            #print(f"關閉socket時發生錯誤: {e}")
            pass
        finally:
            self.socket.close()
    
        # 關閉所有客戶端連線
        for addr, sock in list(self.clients.items()):
            self.close_client(addr)
        
        print("伺服器已停止")
    
    
    # ✅ MQTT 事件處理
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """當 MQTT 連線成功時執行"""
        if rc == 0:
            print("✅ 成功連線到 MQTT Broker")
        else:
            print(f"❌ MQTT 連線失敗，錯誤碼: {rc}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """✅ 當 MQTT 訊息到達時，更新主題數據"""
        topic = msg.topic
        message = msg.payload.decode("utf-8", errors="ignore").strip()
        print(f"📩 MQTT 訂閱主題 [{topic}] 收到: {message}")
        self.subscribed_topics[topic] = message  # ✅ 更新主題最新值

    def get_topic_value(self, topic):
        """✅ 獲取某個主題的最新數值"""
        return self.subscribed_topics.get(topic, "訂閱失敗，請再一次")
        
if __name__ == "__main__":
    server = XiaoZhi_Ai_TCPServer()
    server.start()
