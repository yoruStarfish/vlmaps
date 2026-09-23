# test_ollama.py
from openai import OpenAI

# 1. 這裡就是偷天換日的關鍵！把原本指向 OpenAI 伺服器的網址，改成你本地的 Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama', # Ollama 其實不需要金鑰，但 OpenAI 套件規定這裡不能是空的，所以隨便填
)

print("正在呼叫本地 LLM，請稍候...")

# 2. 發送測試對話
response = client.chat.completions.create(
    model="phi3", # 記得填入你剛剛下載的模型名稱
    messages=[
        {"role": "system", "content": "You are a robot navigation assistant. Output ONLY python code."},
        {"role": "user", "content": "Write a python function to print 'Hello VLMaps'."}
    ],
    temperature=0.0 # 設為 0 讓模型不會亂發揮
)

# 3. 印出結果
print("LLM 回覆內容：\n")
print(response.choices[0].message.content)