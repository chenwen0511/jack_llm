from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from typing import Optional, Callable, List, Tuple, Union
import copy
import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from packaging import version

# 定义错误信息常量，用于提示用户使用正确的模型和配置进行聊天。
_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""
# 定义消息结束符常量
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
# 定义历史记录类型和token序列类型的别名，以方便后续使用
HistoryType = List[Tuple[str, str]] # 每个元素是一个元组，包含用户的输入和模型的回复
TokensType = List[int] # token序列，即文本编码后的整数列表
BatchTokensType = List[List[int]] # 批次token序列，每个元素都是一个token序列

def get_stop_words_ids(chat_format, tokenizer):
    """
    根据指定的聊天格式(chat_format)和分词器(tokenizer)，获取停止词的ID列表。

    参数:
        chat_format (str): 聊天格式，例如"raw"或"chatml"。
        tokenizer (PreTrainedTokenizer): 分词器实例，用于将文本转换为token ID。

    返回:
        List[List[int]]: 停止词的ID列表，每个元素都是一个token ID列表。
    """
    if chat_format == "raw":
        # 如果聊天格式是原始格式，停止词包括"Human:"的token序列和EOD（End of Document）标记的ID。
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        # 如果聊天格式是chatml格式，停止词包括IMEND（Assistant消息结束符）的token ID和IMSTART（开始符）的token ID。
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        # 如果聊天格式不是已知的格式，抛出NotImplementedError异常。
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    """
    准备用于模型生成回复的上下文。

    参数:
        tokenizer (PreTrainedTokenizer): 分词器实例。
        query (str): 当前用户的查询。
        history (List[Tuple[str, str]], optional): 对话历史，每个元素是一个元组，包含用户的输入和模型的回复，默认为空。
        system (str, optional): 系统提示信息，默认为空字符串。
        max_window_size (int, optional): 上下文窗口的最大token数量，默认为6144。
        chat_format (str, optional): 聊天格式，支持"chatml"和"raw"，默认为"chatml"。

    返回:
        Tuple[str, List[int]]: 包含原始文本和对应的token序列。
    """
    if history is None:
        history = []  # 如果没有提供历史，则初始化为空列表

    if chat_format == "chatml":
        im_start_tokens = [tokenizer.im_start_id]  # 获取对话开始符的token ID
        im_end_tokens = [tokenizer.im_end_id]  # 获取对话结束符的token ID
        im_start, im_end = tokenizer.decode(im_start_tokens, skip_special_tokens=False), tokenizer.decode(im_end_tokens, skip_special_tokens=False)  # 解码开始符和结束符为字符串
        nl_tokens = tokenizer.encode("\n")  # 编码换行符为token序列

        def _tokenize_str(role, content):
            """
            将角色和内容组合成字符串并进行token化。

            参数:
                role (str): 角色名称，如"user"或"assistant"。
                content (str): 角色所说的内容。

            返回:
                Tuple[str, List[int]]: 包含原始字符串和对应的token序列。
            """
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())  # 返回组合后的字符串及其token序列

        system_text, system_tokens_part = _tokenize_str("system", system)  # 处理系统提示信息
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens  # 添加开始符和结束符

        raw_text = ""  # 初始化原始文本为空字符串
        context_tokens = []  # 初始化上下文token序列为空列表

        # 反向遍历历史记录，构建上下文
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)  # 处理用户查询
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens  # 添加开始符和结束符
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )  # 处理助手回复
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens  # 添加开始符和结束符

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens  # 构建下一个上下文token序列
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )  # 构建上一次对话的原始文本

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )  # 计算当前上下文的总token数量
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens  # 更新上下文token序列
                raw_text = prev_chat + raw_text  # 更新原始文本
            else:
                break  # 如果超出最大窗口大小，则停止添加历史记录

        context_tokens = system_tokens + context_tokens  # 在上下文token序列前添加系统提示信息的token序列
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text  # 在原始文本前添加系统提示信息
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )  # 添加当前用户的查询和助手开始符
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"  # 在原始文本后添加当前用户的查询和助手开始符

    elif chat_format == "raw":
        raw_text = query  # 如果格式为"raw"，则原始文本就是用户的查询
        context_tokens = tokenizer.encode(raw_text)  # 直接对原始文本进行token化
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")  # 如果格式未知，则抛出异常

    return raw_text, context_tokens  # 返回原始文本和上下文token序列

class vLLMWrapper:
    def __init__(self,
                 model_dir: str,
                 trust_remote_code: bool = True,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.98,
                 dtype: str = "bfloat16",
                 **kwargs):
        """
        初始化vLLMWrapper类。

        参数:
            model_dir (str): 模型目录路径。
            trust_remote_code (bool, optional): 是否信任远程代码，默认为True。
            tensor_parallel_size (int, optional): 张量并行度，默认为1。
            gpu_memory_utilization (float, optional): GPU内存利用率，默认为0.98。
            dtype (str, optional): 数据类型，默认为"bfloat16"。
            **kwargs: 其他关键字参数。
        """
        if dtype not in ("bfloat16", "float16", "float32"):
            print("now not support {}!".format(dtype))
            raise Exception  # 如果数据类型不受支持，则抛出异常

        # 构建生成配置
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

        # 构建分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id  # 设置结束符的token ID

        self.stop_words_ids = []  # 初始化停止词的token ID列表

        from vllm import LLM
        import vllm
        if version.parse(vllm.__version__) >= version.parse("0.2.2"):
            self.__vllm_support_repetition_penalty = True  # 检查是否支持重复惩罚
        else:
            self.__vllm_support_repetition_penalty = False

        quantization = getattr(kwargs, 'quantization', None)  # 获取量化参数

        # 初始化模型
        self.model = LLM(model=model_dir,
                         tokenizer=model_dir,
                         tensor_parallel_size=tensor_parallel_size,
                         trust_remote_code=trust_remote_code,
                         quantization=quantization,
                         gpu_memory_utilization=gpu_memory_utilization,
                         dtype=dtype)

        # 获取停止词的token ID并添加到停止词列表中
        for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

    def chat(self,
             query: str,
             history: Optional[HistoryType],
             tokenizer: PreTrainedTokenizer = None,
             system: str = "You are a helpful assistant.",
             generation_config: Optional[GenerationConfig] = None,
             **kwargs):
        """
        进行多轮对话。

        参数:
            query (str): 用户的查询。
            history (Optional[HistoryType]): 对话历史，默认为None。
            tokenizer (PreTrainedTokenizer, optional): 分词器，默认为None。
            system (str, optional): 系统提示信息，默认为"You are a helpful assistant."。
            generation_config (Optional[GenerationConfig], optional): 生成配置，默认为None。
            **kwargs: 其他关键字参数。
        """
        generation_config = generation_config if generation_config is not None else self.generation_config
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        # 确保生成配置中的聊天格式为"chatml"
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT

        # 检查是否支持重复惩罚
        if not self.__vllm_support_repetition_penalty and generation_config.repetition_penalty != 1:
            raise RuntimeError("The installed vLLM doesn't support repetition_penalty, please set ``model.generation_config.repetition_penalty = 1`` or install vllm>=0.2.2")

        if history is None:
            history = []
        else:
            # 深拷贝用户输入的历史记录，避免修改原始数据
            history = copy.deepcopy(history)

        extra_stop_words_ids = kwargs.get('stop_words_ids', None)
        if extra_stop_words_ids is None:
            extra_stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size

        from vllm.sampling_params import SamplingParams
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "early_stopping": False,
            "top_p": generation_config.top_p,
            "top_k": -1 if generation_config.top_k == 0 else generation_config.top_k,
            "temperature": generation_config.temperature,
            "max_tokens": generation_config.max_new_tokens,
            "repetition_penalty": generation_config.repetition_penalty
        }
        if not self.__vllm_support_repetition_penalty:
            sampling_kwargs.pop("repetition_penalty")  # 如果不支持重复惩罚，移除相关参数
        sampling_params = SamplingParams(**sampling_kwargs)

        # 构建上下文
        raw_text, context_tokens = make_context(
            self.tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        # 生成回复
        req_outputs = self.model.generate([query],
                                            sampling_params=sampling_params,
                                            prompt_token_ids=[context_tokens])
        req_output = req_outputs[0]

        prompt_str = req_output.prompt
        prompt_ids = req_output.prompt_token_ids
        req_sample_output_ids = []
        req_sample_output_strs = []
        for sample in req_output.outputs:
            output_str = sample.text
            output_ids = sample.token_ids
            if IMEND in output_str:
                output_str = output_str[:-len(IMEND)]  # 移除IMEND标记
            if ENDOFTEXT in output_str:
                output_str = output_str[:-len(ENDOFTEXT)]  # 移除ENDOFTEXT标记
            req_sample_output_ids.append(prompt_ids + output_ids)
            req_sample_output_strs.append(prompt_str + output_str)
        assert len(req_sample_output_strs) == 1  # 确保只有一个输出
        response = req_sample_output_strs[0][len(prompt_str):]  # 提取回复部分
        history.append((prompt_str, response))  # 更新对话历史

        return response, history  # 返回回复和更新后的对话历史

if __name__ == '__main__':
    # 主程序入口

    model_dir = 'Qwen/Qwen-72B-Chat'  # 指定模型目录路径
    tensor_parallel_size = 2  # 设置张量并行度

    # 初始化vLLMWrapper模型
    model = vLLMWrapper(model_dir,
                        tensor_parallel_size=tensor_parallel_size,
                        )

    response, history = model.chat(query="你好",
                                   history=None)  # 初始化对话，没有历史记录
    print(response)  # 打印模型的回复

    # 第二次对话
    response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。",
                                   history=history)  # 使用上次的对话历史
    print(response)  # 打印模型的回复

    # 第三次对话
    response, history = model.chat(query="给这个故事起一个标题",
                                   history=history)  # 继续使用上次的对话历史
    print(response)  # 打印模型的回复
