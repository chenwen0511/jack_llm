import argparse
import json
from typing import Dict
import logging

import torch
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# 定义一个常量 IGNORE_TOKEN_ID，用于在目标序列中表示忽略的标记
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources, # 原始对话数据列表
    tokenizer: transformers.PreTrainedTokenizer,# 使用的分词器
    max_len: int,# 输入的最大长度
    system_message: str = "You are a helpful assistant."
) -> Dict:
    """
    预处理对话数据，将其转换为模型可以理解的格式。

    参数:
    - sources: 包含对话数据的列表。
    - tokenizer: 用于分词的预训练分词器。
    - max_len: 输入序列的最大长度。
    - system_message: 系统消息的文本，默认为 "You are a helpful assistant."。

    返回:
    - 一个包含预处理后数据的列表，每个元素是一个包含输入ID和注意力掩码的字典。
    """

    # 定义不同角色的前缀
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    # 获取特殊标记的ID
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids  # 获取换行符的输入ID

    # 将系统、用户和助手的标记转换为输入ID
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # 初始化预处理后的数据列表
    data = []

    # 遍历每个对话
    for i, source in enumerate(sources):
        source = source["conversations"]
        # 如果第一个元素不是用户，跳过它
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        # 初始化当前对话的输入ID和目标ID
        input_id, target = [], []
        # 构建系统消息的输入ID和目标ID
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        # 确保输入ID和目标ID的长度相同
        assert len(input_id) == len(target)
        # 遍历对话中的每个句子
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # 构建当前句子的输入ID
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            # 根据角色构建目标ID
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        # 确保输入ID和目标ID的长度相同
        assert len(input_id) == len(target)
        # 将输入ID转换为张量，并截断到最大长度
        input_id = torch.tensor(input_id[:max_len], dtype=torch.int)
        # 计算注意力掩码，非填充符的位置为1
        attention_mask = input_id.ne(tokenizer.pad_token_id)
        # 将预处理后的数据添加到列表中
        data.append(dict(input_ids=input_id, attention_mask=attention_mask))

    # 返回处理后的数据列表
    return data


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("Model Quantization using AutoGPTQ")
    # 添加命令行参数
    parser.add_argument("--model_name_or_path", type=str, help="model path")
    parser.add_argument("--data_path", type=str, help="calibration data path")
    parser.add_argument("--out_path", type=str, help="output path of the quantized model")
    parser.add_argument("--max_len", type=int, default=8192, help="max length of calibration data")
    parser.add_argument("--bits", type=int, default=4, help="the bits of quantized model. 4 indicates int4 models.")
    parser.add_argument("--group-size", type=int, default=128, help="the group size of quantized model")
    # 解析命令行参数
    args = parser.parse_args()

    # 创建量化配置
    quantize_config = BaseQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,# 指定量化时使用的组大小。组量化是一种技术，它将模型中的多个权重组合在一起进行量化，以减少模型大小并提高计算效率
        damp_percent=0.01, #用于在量化过程中控制权重的调整程度。较高的值可以减少量化带来的影响
        desc_act=False,  # 设置为 False 可以显著加快推理速度，但困惑度可能会略有下降
        static_groups=False, # 是否在量化过程中使用静态量化。如果设置为 True，则在量化过程中组不会被动态调整
        sym=True, # 对称性。控制量化是否是对称的，可以减少量化误差
        true_sequential=True, # 控制量化过程中是否考虑权重的顺序。如果设置为 True，则量化过程会考虑权重的顺序，这可能会提高量化后的模型精度
        model_name_or_path=None,
        model_file_base_name="model"
    )

    # 从预训练模型路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 设置分词器的填充符ID为结束符ID
    tokenizer.pad_token_id = tokenizer.eod_id

    # 使用预处理函数处理数据
    data = preprocess(json.load(open(args.data_path)), tokenizer, args.max_len)

    # 从预训练模型路径加载模型，并应用量化配置
    model = AutoGPTQForCausalLM.from_pretrained(args.model_name_or_path, quantize_config, device_map="auto", trust_remote_code=True)

    # 设置日志记录配置
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    # 使用处理后的数据对模型进行量化
    model.quantize(data, cache_examples_on_gpu=False)

    # 保存量化后的模型到指定路径
    model.save_quantized(args.out_path, use_safetensors=True)
    # 同时保存分词器文件到同一路径
    tokenizer.save_pretrained(args.out_path)
