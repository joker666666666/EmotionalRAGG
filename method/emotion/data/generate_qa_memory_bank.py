# 1 加载文件
import json
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

from utils.config import OPENAI_API_KEY, OPENAI_API_BASE


def call_llm(system_prompt, user_input):  # GPT-3.5 Turbo的价格是 $0.50 / 1M tokens, GPT-4 Turbo的价格是 $10 / 1M tokens
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
 
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user", 
                "content": user_input
            },
        ],
    )

    content = completion.choices[0].message.content
    usage = completion.usage
    return content, usage


def str2vector(text):
    match = re.findall(r"【(.*?)】", text)
    emotion_str = match[-1]

    emotion_str = emotion_str.replace(": ", "：")
    emotion_str = emotion_str.replace(":", "：")
    emotion_str = emotion_str.replace(", ", "，")
    emotion_str = emotion_str.replace(",", "，")

    emotion_list = emotion_str.split("，")
    emotion_embedding = [float(e.split("：")[1]) for e in emotion_list]
    return emotion_embedding


def process_data(role_data_dict):
    emotions = ["joy", "acceptance", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]


    response, _ = call_llm(system_prompt.format(role=role_data_dict['role']), role_data_dict["context"])
    try:
        emotion_list = json.loads(response)
    except:
        # print(response)
        return None
    emotion_embedding = [1, 1, 1, 1, 1, 1, 1, 1]
    for item in emotion_list:
        try:
            if item["dim"] in emotions:
                emotion_embedding[emotions.index(item["dim"])] = item["score"]
        except:
            print(item)
    result = {
        "context": role_data_dict["context"],
        "emotion_embedding": emotion_embedding,
        "context_embedding": role_data_dict["context_embedding"]
    }
    return result


system_prompt = """
你是一个情感分析大师，你能够仔细分辨每段对话角色的情绪状态。
假设每个角色共有8种基本情绪，包括 joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation 。
接下来我会输入一段{role}的对话，你的任务是分析{role}在这8个情绪维度上的得分，最低为1分，最高为10分，得分越高表明{role}在这个情绪维度上表达越强烈。
请分析{role}在8个情绪维度上的表现，给出打分理由和得分，最后以 python list 的形式输出结果，如下所示：
[
    {{"analysis": <REASON>, "dim": "joy", "score": <SCORE>}},
    {{"analysis": <REASON>, "dim": "acceptance", "score": <SCORE>}},
    ...
    {{"analysis": <REASON>, "dim": "anticipation", "score": <SCORE>}}
]
你的回答必须是一个有效的 python 列表以保证我能够直接使用 python 解析它，不要有多余的内容！请给出尽可能准确的、符合大多数人直觉的结果。
"""

with open("/home/huanglebj/EmotionMEM/data/CharacterLLM/characterllm_memory_bank.json", "r", encoding="utf-8") as f:
    datas = json.load(f)

all_bank = {}
for role, data in datas.items():
    print(role)
    data = data[:1000]
    for d in data:
        d["role"] = role

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(process_data, data), total=len(data)))
    results = [r for r in results if r is not None]
    all_bank[role] = results
    # break

with open("/home/huanglebj/EmotionMEM/method/emotion/data/characterllm_memory_bank.jsonl", "w", encoding="utf-8") as f:
    json.dump(all_bank, f, ensure_ascii=False, indent=2)

with open("/home/huanglebj/EmotionMEM/method/emotion/data/characterllm_memory_bank.jsonl", "r", encoding="utf-8") as f:
    datas = json.load(f)
print(datas.keys())
print("保存成功")