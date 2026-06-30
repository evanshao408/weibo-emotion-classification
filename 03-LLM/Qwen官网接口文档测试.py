from openai import OpenAI    # pip install openai
import os
from dotenv import load_dotenv

load_dotenv('.env_Qwen')

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen3.6-flash",
    messages=[{'role': 'user', 'content': '你是谁？'}],
    stream=False
)

print(completion.choices[0].message.content)

