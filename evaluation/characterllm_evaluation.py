import argparse
import concurrent.futures
import json
import logging
import re
import threading
from datetime import datetime
from functools import partial

import pytz
from tqdm import tqdm

from utils.characterllm_evaluation_prompt import (hallucination_prompt,
                                                  memorization_prompt,
                                                  personality_prompt,
                                                  values_prompt)
from utils.functions import call_llm


def process_data(data, prompt):
    global completion_tokens, prompt_tokens

    # role = data["role"]
    # role_information_json = role_informations[role]
    # role_information = "\n".join(f"{key}：{value}" for key, value in role_information_json.items())
    role = "Spartacus"

    system_prompt = prompt.format(role=role, role_information=role_information)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": data["model_output"]},
    ]
    response, usage = call_llm("gpt-3.5-turbo", messages)
    
    with lock:
        completion_tokens += usage.completion_tokens
        prompt_tokens += usage.prompt_tokens

    try:
        scores = re.findall(r"\d+", response)
        return int(scores[-1])
    except:
        return None


def evaluate_scores(prompt, prompt_name, datas):
    process_data_with_prompt = partial(process_data, prompt=prompt)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        scores = list(tqdm(executor.map(process_data_with_prompt, datas), total=len(datas)))
    scores = [score for score in scores if score is not None]
    if scores:
        average_score = sum(scores) / len(scores)
        logger.info(f"{prompt_name}: {average_score}")
    else:
        logger.info(f"{prompt_name}: No scores to average")


lock = threading.Lock()
completion_tokens = 0
prompt_tokens = 0
# with open("data/CharacterEval/character_profiles.json", "r", encoding="utf-8") as f:
#     role_informations = json.load(f)
with open("data/CharacterLLM/profiles/wiki_Spartacus.txt", "r", encoding="utf-8") as f:
    role_information = f.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="method/original/result/generation1.json")
    parser.add_argument("--log_path", type=str, default="evaluation/log/CharacterLLM/result.log")
    args = parser.parse_args()

    # 创建一个 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建一个 handler，用于写入日志文件
    fh = logging.FileHandler(args.log_path)
    fh.setLevel(logging.INFO)

    # 创建一个 handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 定义 handler 的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给 logger 添加 handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 使用北京时间的时区
    beijing_tz = pytz.timezone('Asia/Shanghai')
    logging.Formatter.converter = lambda *args: datetime.now(tz=beijing_tz).timetuple()
    
    with open(args.result_path, "r", encoding="utf-8") as f:
        original_datas = json.load(f)

    evaluate_scores(memorization_prompt, "Memorization", original_datas)
    evaluate_scores(personality_prompt, "Personality", original_datas)
    evaluate_scores(values_prompt, "Values", original_datas)
    evaluate_scores(hallucination_prompt, "Hallucination", original_datas)

    logger.info(f"Completion tokens: {completion_tokens}, Prompt tokens: {prompt_tokens}")