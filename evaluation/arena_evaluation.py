import concurrent.futures
import json
import time
from collections import Counter

from openai import OpenAI
from tqdm import tqdm
from utils.config import OPENAI_API_BASE, OPENAI_API_KEY


def call_llm(model, system_prompt, input):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input},
    ]

    completion = client.chat.completions.create(model=model, messages=messages)

    content = completion.choices[0].message.content
    usage = completion.usage
    return content, usage


def process_pair(d_1, d_2, d_3):
    result = []
    if d_1["id"] == d_2["id"] and d_1["id"] == d_2["id"]:
        role = d_1["role"]
        role_profile = character_profiles[role]
        role_information = "\n".join(
            f"{key}：{value}" for key, value in role_profile.items()
        )

        input = eval_prompt.format(
            role_name=role,
            character_profile=role_information,
            context=d_1["context"],
            answer={
                "MODEL_1": d_1["model_output"],
                "MODEL_2": d_2["model_output"],
                "MODEL_3": d_3["model_output"],
            },
        )

        system_prompt = "你是一个角色扮演的效果对比助手，你会根据输出的角色特征和质量来对模型进行排名，然后使用 JSON 的数据格式输出结果。"
        response, usage = call_llm("gpt-4-turbo", system_prompt, input)

        try:
            result = json.loads(response)
        except:
            print(response)

    return result


eval_prompt = """下列模型要扮演的角色是{role_name}。{role_name}的角色描述是：
{character_profile}

你需要根据下面两个原则对下列模型进行排名：
1. 哪一个的角色说话风格特征更加明显,说话更加符合角色描述,说话越有特色就越好;
2. 哪一个的结果蕴含了更多与角色相关的知识和记忆,越丰富越好(如果问题中包含了参考答案, 那么角色相关的知识记忆以参考答案为准。);

输入给各个模型的上下文是： 
{context} 

各个模型针对该上下文的回复分别为： 
{answer} 

现在请你根据上述两个原则，对各个模型进行排名。避免任何位置偏见，并确保模型回答的呈现顺序不会影响你的决定。
不要对模型的名字带有偏见。然后使用一个包含模型与其排名、这样排名的理由的列表返回结果，也就是说，请务必使用如下格式返回结果： 
[
    {{
        "model_name": model-name, 
        "rank_reason": rank-reason, 
        "model_rank": model-rank
    }},
    {{
        "model_name": model-name, 
        "rank_reason": rank-reason, 
        "model_rank": model-rank
    }},
    {{
        "model_name": model-name, 
        "rank_reason": rank-reason, 
        "model_rank": model-rank
    }}
] 
你的回答必须是一个有效的 JSON 对象以保证我能够直接使用 python 解析它，不要有多余的内容！请给出尽可能准确的、符合大多数人直觉的排名。
"""

with open("original/result/generation1.jsonl", "r", encoding="utf-8") as f:
    original_datas = json.load(f)
with open("memory/result/generation2.jsonl", "r", encoding="utf-8") as f:
    memory_datas = json.load(f)
with open("emotion/result/generation3.jsonl", "r", encoding="utf-8") as f:
    emotion_datas = json.load(f)
with open("data/character_profiles.json", "r", encoding="utf-8") as f:
    character_profiles = json.load(f)

max = 100
model_1_config = {"model_name": "emotion", "datas": emotion_datas[:max]}
model_2_config = {"model_name": "memory", "datas": memory_datas[:max]}
model_3_config = {"model_name": "original", "datas": original_datas[:max]}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_to_result = {
        executor.submit(process_pair, d_1, d_2, d_3): (d_1, d_2, d_3)
        for d_1, d_2, d_3 in zip(
            model_1_config["datas"], model_2_config["datas"], model_3_config["datas"]
        )
    }
    results = []
    for future in tqdm(concurrent.futures.as_completed(future_to_result), total=max):
        results.extend(future.result())

win_model = []
for data in results:
    try:
        if data["model_rank"] == 1:
            win_model.append(data["model_name"])
    except:
        print(data)

print(Counter(win_model))

time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
save_path = f"evaluation/log/Arena/{model_1_config['model_name']}_vs_{model_2_config['model_name']}_vs_{model_3_config['model_name']}_{max}samples_{time_stamp}.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
