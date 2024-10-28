# -*- encoding: utf-8 -*-
"""
@File    :   llm.py
@Time    :   2024/05/21 13:46:42
@Author  :   LanSnowZ 
@Version :   1.0
@Contact :   lansnowzzz@gmail.com
@Desc    :   None
"""

import threading
from typing import Any, List

import torch
from anthropic import Anthropic
from openai import OpenAI
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_local_model(model_path: str = ""):
    if "Qwen1.5" in model_path:
        return ChatQwen(model_path)
    elif "chatglm" in model_path:
        return ChatGLM(model_path)
    elif "yi" in model_path:
        return ChatYi(model_path)
    raise ValueError("Model not found!")


def load_online_model(model_name: str = "", api_key: str = "", base_url: str = ""):
    if "gpt" in model_name:
        return ChatOpenAI(api_key, base_url, model_name)
    elif "claude" in model_name:
        return ChatAnthropic(api_key, base_url, model_name)
    raise ValueError("Model not found!")


class SingletonMeta(type):
    _instance_lock = threading.Lock()
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._instance_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ChatQwen(metaclass=SingletonMeta):
    """
    Chat model based on Qwen1.5.
    """

    def __init__(self, model_path: str = "/share/base_model/Qwen1.5-7B-Chat") -> None:
        """
        Load and init model.
        """
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=False, trust_remote_code=True
        )

        # 下面这段同时适用于单卡和多卡
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()

    def query(self, messages: List[dict]) -> str:
        """
        Chat with the model.

        Args:
            messages (List[dict]): messages include prompt and history.

        Returns:
            str: model response
        """
        torch.cuda.empty_cache()
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # [text1, text2..] can be extended to batch inputs
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_k=5,
        )
        # batch decode
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response

    def get_token_num(self) -> tuple:
        """
        Get token number of input and output.

        Returns:
            tuple: (prompt_token_num, completion_token_num)
        """
        if self.__token_num_valid is False:
            raise ValueError(
                "Please call chat first or wait for the finish of generation!"
            )
        return self.__prompt_token_num, self.__completion_token_num


class ChatOpenAI(object):
    """
    Chat model based on OpenAI.
    """

    def __init__(
        self, api_key: str = "", base_url: str = "", model_name: str = ""
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.chat_client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def query(self, messages: List[dict]) -> str:
        response = self.chat_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=0.8,
            top_p=0.5,
        )

        return response.choices[0].message.content


class ChatAnthropic(object):
    """
    Chat model based on AnthropicAPI.
    """

    def __init__(
        self, api_key: str = "", base_url: str = "", model_name: str = ""
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.chat_client = Anthropic(api_key=self.api_key, base_url=self.base_url)

    def query(self, messages: List[dict]) -> str:
        response = self.chat_client.messages.create(
            max_tokens=2048,
            system=messages[0]["content"],
            messages=messages[1:],
            model=self.model_name,
            stream=False,
            temperature=0.8,
            top_p=0.5,
        )

        return response.content[0].text


class ChatGLM(metaclass=SingletonMeta):
    def __init__(self, model_path: str = "/share/base_model/chatglm3-6b") -> None:
        """
        Load and init model.
        """
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True)
            .half()
            .cuda()
            .eval()
        )

    def query(self, messages: List[dict]) -> str:
        # TODO: handle system prompt(GLM not support)
        """
        Chat with the model.

        Args:
            messages (List[dict]): messages include prompt and history.

        Returns:
            str: model response
        """
        response, _ = self.model.chat(self.tokenizer, messages[-1], messages[:-1])
        return response


class ChatYi(metaclass=SingletonMeta):
    def __init__(self, model_path: str = "/share/base_model/Yi-6B-Chat") -> None:
        """
        Load and init model.
        """
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype="auto"
        ).eval()

    def query(self, messages: List[dict]) -> str:
        # TODO: handle system prompt(Yi not support)
        """
        Chat with the model.

        Args:
            messages (List[dict]): messages include prompt and history.

        Returns:
            str: model response
        """
        model_inputs = self.tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, return_tensors="pt"
        ).to("cuda")

        output_ids = self.model.generate(
            model_inputs, eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(
            output_ids[0][model_inputs.shape[1] :], skip_special_tokens=True
        )

        return response
