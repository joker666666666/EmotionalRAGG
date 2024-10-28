#!/bin/bash

CUR_TIME=$(TZ='Asia/Shanghai' date +%Y%m%d-%H%M%S)
RESULT_PATH=memory/result/${CUR_TIME}.jsonl

python memory/get_response.py \
    --model_name "Qwen" \
    --device "cuda:1" \
    --example_number 100 \
    --result_path ${RESULT_PATH}  &&

LOG_PATH=memory/log/${CUR_TIME}.log

python memory/CharacterLLM_evaluator.py \
    --result_path ${RESULT_PATH} \
    --log_path ${LOG_PATH}