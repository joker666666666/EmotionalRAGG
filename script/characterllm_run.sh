#!/bin/bash

CUR_TIME=$(TZ='Asia/Shanghai' date +%Y%m%d-%H%M%S)
MODEL=gpt
ROLE=Spartacus
METHOD=memory
RESULT_PATH=method/${METHOD}/result/CharacterLLM_${ROLE}_${MODEL}_${CUR_TIME}.jsonl

python method/${METHOD}/get_response.py \
    --model_name ${MODEL} \
    --device "cuda:1" \
    --result_path ${RESULT_PATH}  &&

LOG_PATH=evaluation/log/CharacterLLM/${ROLE}_${MODEL}_${CUR_TIME}.log

python evaluation/characterllm_evaluation.py \
    --result_path ${RESULT_PATH} \
    --log_path ${LOG_PATH}