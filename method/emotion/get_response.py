import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

sys.path.append("/home/huanglebj/EmotionMEM")

from utils.functions import (call_chatglm, call_chatglm2, call_gpt, call_gpt2,
                             call_llm, call_qwen, call_qwen2,
                             load_model_and_tokenizer)


def retrieval(query, method="C-A"):
    role = query["role"]
    context_embedding = query["context_embedding"]
    emotion_embedding = query["emotion_embedding"]

    context_distances = [np.linalg.norm(np.array(context_embedding) - np.array(mem["context_embedding"])) for mem in memory_bank[role]]
    emotion_distances = [distance.cosine(np.array(emotion_embedding), np.array(mem["emotion_embedding"])) for mem in memory_bank[role]]

    if method == "C-M":
        combine_distances = [a * b for a, b in zip(context_distances, emotion_distances)]
        inx = np.argsort(combine_distances)[:10] 
    elif method == "C-A":
        combine_distances = [a + b for a, b in zip(context_distances, emotion_distances)]
        inx = np.argsort(combine_distances)[:10] 
    elif method == "S-C":
        context_indices = np.argsort(context_distances)[:20]
        result_1 = [emotion_distances[i] for i in context_indices]
        emotion_indices = np.argsort(result_1)[:10]
        inx = [context_indices[i] for i in emotion_indices]
    elif method == "S-S":
        emotion_indices = np.argsort(emotion_distances)[:20]
        result_1 = [context_distances[i] for i in emotion_indices]
        context_indices = np.argsort(result_1)[:10]
        inx = [emotion_indices[i] for i in context_indices]
    elif method == "OriginalRAG":
        inx = np.argsort(context_distances)[:10]
    else:
        raise ValueError("Invalid method")

    nearest_docs = [memory_bank[role][i]["context"] for i in inx]
    scores = [context_distances[i] for i in inx]
    return nearest_docs, scores


def get_response_characterllm(query, model_name, model, tokenizer, device, retrieval_method):
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
Interviewer (speaking): {query}
{character} (speaking): """
    character = query["role"]
    loc_time = "Coffee Shop - Afternoon"
    status = f"{character} is casually chatting with a man from the 21st century. {character} fully trusts the man who engage in conversation and shares everything {character} knows without reservation."
    interviewer = "Man"
    related_dialogues, _ = retrieval(query, retrieval_method)
    concat_dialogues = "\n\n".join(related_dialogues)
    role_system = role_system.format(
        character=character,
        loc_time=loc_time,
        status=status,
        query=query["context"],
        related_dialogues=concat_dialogues,
    )

    if model_name == "qwen":
        response = call_qwen2(character, role_system, model, tokenizer, device)
    elif model_name == "chatglm":
        response = call_chatglm2(character, role_system, model, tokenizer)
    elif model_name == "gpt-3.5":
        response = call_gpt2(character, role_system)

    query["model_output"] = response
    return query


def get_response_charactereval(query, model_name, model, tokenizer, device, retrieval_method):
    context = query["context"]
    role = query["role"]

    role_information_json = role_informations[role]
    role_information = "\n".join(
        f"{key}：{value}" for key, value in role_information_json.items()
    )

    related_dialogues, related_score = retrieval(query, retrieval_method)
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
        f"现在你是{role}，请你仿照{role}的语气和说话方式，参考角色信息和记忆内容，回答采访者的问题。请不要脱离角色，绝对不要说自己是人工智能助手。\n"
        f"采访者：{context}"
    )

    if model_name == "qwen":
        response = call_qwen2(role, role_system, model, tokenizer, device)
    elif model_name == "chatglm":
        response = call_chatglm2(role, role_system, model, tokenizer)
    elif model_name == "gpt-3.5":
        response = call_gpt2(role, role_system)

    query["model_output"] = response
    return query


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="characterllm")
    parser.add_argument("--model_name", type=str, default="gpt-3.5")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--retrieval_method", type=str, default="C-A")
    args = parser.parse_args()

    if args.dataset == "charactereval":
        global role_informations
        with open("data/CharacterEval/character_profiles.json", "r", encoding="utf-8") as f:
            role_informations = json.load(f)

    global memory_bank
    with open(f"method/emotion/data/{args.dataset}/memory_bank.jsonl", "r", encoding="utf-8") as f:
        memory_bank = json.load(f)

    with open(f"method/emotion/data/{args.dataset}/MBTI_label.json", "r", encoding="utf-8") as f:
        chatacter_data = json.load(f)
    character_roles = list(chatacter_data.keys())

    global query_bank
    if args.dataset == "characterllm":  #  English dataset
        with open("method/emotion/data/incharacter/query_bank_16P_en.jsonl", "r", encoding="utf-8") as f:
            query_bank = json.load(f)
    else:  # Chinese dataset
        with open(f"method/emotion/data/incharacter/query_bank_16P_zh.jsonl", "r", encoding="utf-8") as f:
            query_bank = json.load(f)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    for i, character in enumerate(character_roles, 1):
        results = []
        for query in tqdm(query_bank, desc=f"第{i}个角色{character}，总共{len(character_roles)}个角色"):
            query['role'] = character
            if args.dataset == "charactereval":
                response = get_response_charactereval(query, args.model_name, model, tokenizer, args.device, args.retrieval_method)
            else:
                response = get_response_characterllm(query, args.model_name, model, tokenizer, args.device, args.retrieval_method)
            results.append(
                {
                    "id": response["id"],
                    "question": response["context"],
                    "response_open": response["model_output"],
                    "query_style": "interview"
                }
            )
        directory_path = f"method/emotion/result/{args.dataset}/{args.retrieval_method}/{args.model_name}"
        os.makedirs(directory_path, exist_ok=True)
        with open(f"{directory_path}/{character}_zh_0_test.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
    print(f"dataset: {args.dataset}, model_name: {args.model_name}, retrieval_method: {args.retrieval_method}")
