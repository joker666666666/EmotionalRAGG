import json
import re
from concurrent.futures import ThreadPoolExecutor

from FlagEmbedding import FlagModel
from openai import OpenAI
from tqdm import tqdm


def call_llm(system_prompt, user_input):  # GPT-3.5 Turbo的价格是 $0.50 / 1M tokens, GPT-4 Turbo的价格是 $10 / 1M tokens
    client = OpenAI(api_key="sk-eYL6X2jG2af7f134f3c3T3BlBkFJ4Ec84A636f314414b8cC", base_url="https://cn2us02.opapi.win/v1")
 
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


def get_result(data):
    emotions = ["joy", "acceptance", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

    lines = data["context"].split("\n")
    query = lines[-1]
    context_embedding = embedding_model.encode(query).tolist()

    # 使用整段context生成embedding
    response, _ = call_llm(system_prompt.format(role=data["role"]), data["context"])
    try:
        emotion_list = json.loads(response)
    except:
        print(response)
        return None
    emotion_embedding = [1, 1, 1, 1, 1, 1, 1, 1]
    for item in emotion_list:
        try:
            if item["dim"] in emotions:
                emotion_embedding[emotions.index(item["dim"])] = item["score"]
        except:
            print(item)
    result = {
        "query": query,
        "gpt_output": response,
        "emotion_embedding": emotion_embedding,
        "context_embedding": context_embedding
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

embedding_model = FlagModel(
    "/share/base_model/bge-base-zh-v1.5",
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True,
)

with open("data/test_data.jsonl", "r", encoding="utf-8") as f:
    datas = json.load(f)

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(get_result, datas), total=len(datas)))

with open("emotion/data/query_bank.jsonl", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open("emotion/data/query_bank.jsonl", "r", encoding="utf-8") as f:
    result = json.load(f)   
print(len(result))
print("保存成功")