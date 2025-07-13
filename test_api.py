import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 测试多个可能的模型
models = [
    "SiliconFlow/Qwen2-7B-Instruct",
    "SiliconFlow/GLM4-9B-Chat",
    "SiliconFlow/DeepSeek-7B-Chat",
    "Qwen2-7B-Instruct"  # 有时不需要前缀
]

for model in models:
    print(f"\n测试模型: {model}")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )

        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(f"响应内容: {response.json()}")
            print(f"成功模型: {model}")
            break
        else:
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"请求失败: {str(e)}")