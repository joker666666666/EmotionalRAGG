import argparse
import json
import numpy as np

from tqdm import tqdm

from utils.functions import (
    call_chatglm,
    call_gpt,
    call_qwen,
    call_llm,
    load_model_and_tokenizer,
)


def retrieval(role, data):
    context = data["context"]
    lines = context.split("\n")
    query_context_embedding = query_bank[lines[-1]]["context_embedding"]

    # 欧几里得距离
    distances = [
        np.linalg.norm(
            np.array(query_context_embedding) - np.array(mem["context_embedding"])
        )
        for mem in memory_bank[role]
    ]

    # 距离最大的10个文档
    inx = np.argsort(distances)[-10:]

    nearest_docs = [memory_bank[role][i]["context"] for i in inx]
    scores = [distances[i] for i in inx]

    return nearest_docs, scores


def retrieval_characterllm(query):
    query_context_embedding = query["context_embedding"]

    # 欧几里得距离
    distances = [
        np.linalg.norm(
            np.array(query_context_embedding) - np.array(mem["context_embedding"])
        )
        for mem in memory_bank
    ]

    # 距离最大的10个文档
    inx = np.argsort(distances)[-10:]

    nearest_docs = [memory_bank[i]["context"] for i in inx]
    scores = [distances[i] for i in inx]

    return nearest_docs, scores


def get_response_characterllm(query, model_name, model, tokenizer, device):
    role_system = """I want you to act like {character}. I want you to respond and answer like {character}, using the tone, manner and vocabulary {character} would use. You must know all of the knowledge of {character}. Reply must be brief and concise.

The status of you is as follows:
Location: {loc_time}
Status: {status}

Example output:
Character1 (speaking): Detailed utterance ...
Character2 (speaking): Detailed utterance ...

Related dialogues:
{related_dialogues}

The conversation begins:
{interviewer} (speaking): {query}"""
    character = "Spartacus"
    loc_time = "Coffee Shop - Afternoon"
    status = f"{character} is casually chatting with a man from the 21st century. {character} fully trusts the man who engage in conversation and shares everything {character} knows without reservation."
    interviewer = "Man"
    related_dialogues, _ = retrieval_characterllm(query)
    concat_dialogues = "\n\n".join(related_dialogues)
    role_system = role_system.format(
        character=character,
        loc_time=loc_time,
        status=status,
        interviewer=interviewer,
        query=query["question"],
        related_dialogues=concat_dialogues,
    )

    if model_name == "Qwen":
        response = call_qwen(role, role_system, model, tokenizer, device)
    elif model_name == "chatglm":
        response = call_chatglm(role, role_system, context, model, tokenizer)
    elif model_name == "gpt":
        # response = call_gpt(role, role_system, context)

        messages = [{"role": "system", "content": role_system}]
        response, usage = call_llm("gpt-3.5-turbo", messages)

    query["model_output"] = response
    query.pop("context_embedding", None)

    return query


def get_response(data, model_name, model, tokenizer, device):
    context = data["context"]
    role = data["role"]

    role_information_json = role_informations[role]
    role_information = "\n".join(
        f"{key}：{value}" for key, value in role_information_json.items()
    )

    related_dialogues, related_score = retrieval(role, data)
    concat_dialogues = "\n\n".join(related_dialogues)

    role_system = (
        f"【角色信息】\n"
        f"---\n"
        f"{role_information}\n"
        f"---\n\n"
        f"【记忆内容】\n"
        f"---\n"
        f"{concat_dialogues}\n"
        f"---\n\n"
        f"角色信息包含{role}的一些基本信息。\n"
        f"记忆内容是{role}回忆起的和当前问题相关的内容。\n"
        f"现在你是一个角色扮演专家，请你遵循角色信息中的人物设定，参考记忆内容中的说话风格和内容，完成对话。\n"
    )

    if model_name == "Qwen":
        response = call_qwen(role, role_system, context, model, tokenizer, device)
    elif model_name == "chatglm":
        response = call_chatglm(role, role_system, context, model, tokenizer)
    elif model_name == "gpt":
        response = call_gpt(role, role_system, context)

    data["related_dialogues"] = related_dialogues
    data["related_score"] = related_score
    data["model_output"] = response
    return data


# with open("data/test_data.jsonl", "r", encoding="utf-8") as f:
#     datas = json.load(f)
# with open("data/character_profiles.json", "r", encoding="utf-8") as f:
#     role_informations = json.load(f)
# with open("memory/data/memory_bank.jsonl", "r", encoding="utf-8") as f:
#     memory_bank = json.load(f)
# with open("memory/data/query_bank.jsonl", "r", encoding="utf-8") as f:
#     query_bank = json.load(f)
with open("data/CharacterLLM/memory_bank_Spartacus.json", "r", encoding="utf-8") as f:
    memory_bank = json.load(f)
with open("data/CharacterLLM/query_bank_Spartacus.json", "r", encoding="utf-8") as f:
    query_bank = json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--example_number", type=int, default=400)
    parser.add_argument(
        "--result_path", type=str, default="memory/result/generation2.jsonl"
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)

    print("开始运行memory/get_response.py")
    results = []
    for query in tqdm(
        query_bank[: min(args.example_number, len(query_bank))], desc="Processing data"
    ):
        results.append(
            get_response_characterllm(
                query, args.model_name, model, tokenizer, args.device
            )
        )

    with open(args.result_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
