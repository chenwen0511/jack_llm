# 导入dataclasses模块，用于定义数据类
from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
# 导入DeepSpeed库，用于大规模深度学习训练的优化
from deepspeed import zero
# 导入DeepSpeed的ZeroParamStatus类，用于管理ZeRO优化器中的参数状态
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
# 导入Transformers库中的Trainer类，用于训练模型
from transformers import Trainer
# 导入Transformers库中的GPTQConfig类，用于配置量化设置
from transformers import GPTQConfig, deepspeed
# 导入Transformers库中的LabelSmoother类，用于平滑标签
from transformers.trainer_pt_utils import LabelSmoother
# 导入PEFT库，用于低秩适应（LoRA）技术
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# 导入Accelerate库的DistributedType类，用于分布式训练
from accelerate.utils import DistributedType

# 定义一个常量IGNORE_TOKEN_ID，用于表示忽略的标签ID
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# 使用dataclass装饰器定义ModelArguments类，用于存储模型参数
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")

# 使用dataclass装饰器定义DataArguments类，用于存储数据参数
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False  # 是否使用懒惰预处理，默认为False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,  # 默认缓存目录为None
        metadata={"help": "Directory to store the cached files."}  # 参数说明
    )
    optim: str = field(
        default="adamw_torch",  # 默认优化器为adamw_torch
        metadata={"help": "Optimizer to use."}  # 参数说明
    )
    model_max_length: int = field(
        default=8192,  # 默认最大序列长度为8192
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False  # 是否使用LoRA进行微调，默认为False

# 使用dataclass装饰器定义LoraArguments类，用于存储LoRA参数
@dataclass
class LoraArguments:
    lora_r: int = 64  # LoRA秩，默认为64
    lora_alpha: int = 16  # LoRA学习率缩放因子，默认为16
    lora_dropout: float = 0.05  # LoRA层的dropout概率，默认为0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"],  # 应用LoRA的目标模块名称，默认为["c_attn", "c_proj", "w1", "w2"]
        metadata={"help": "Names of the modules to apply LoRA to."}  # 参数说明
    )
    lora_weight_path: str = ""  # LoRA权重路径，默认为空字符串
    lora_bias: str = "none"  # LoRA的bias类型，默认为"none"
    q_lora: bool = False  # 是否使用QLoRA，默认为False

# 定义maybe_zero_3函数，用于处理DeepSpeed ZeRO-3优化器中的参数,确保参数可以从GPU复制到CPU并克隆。
def maybe_zero_3(param):
    if hasattr(param, "ds_id"):  # 检查参数是否有ds_id属性
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE  # 确保参数状态为NOT_AVAILABLE
        with zero.GatheredParameters([param]):  # 使用GatheredParameters上下文管理器
            param = param.data.detach().cpu().clone()  # 将参数从GPU复制到CPU并克隆
    else:
        param = param.detach().cpu().clone()  # 如果没有ds_id属性，直接将参数从GPU复制到CPU并克隆
    return param  # 返回处理后的参数

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    # 根据bias参数的不同值，筛选出需要保存的LoRA参数
    if bias == "none":
        # 如果bias为"none"，只保存包含"lora_"前缀的参数
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        # 如果bias为"all"，保存包含"lora_"前缀的参数和所有包含"bias"的参数
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        # 如果bias为"lora_only"，保存包含"lora_"前缀的参数及其对应的bias参数
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t  # 保存包含"lora_"前缀的参数
                bias_name = k.split("lora_")[0] + "bias"  # 生成对应的bias名称
                lora_bias_names.add(bias_name)  # 将bias名称添加到集合中
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError  # 如果bias参数值不合法，抛出异常

    # 使用maybe_zero_3函数处理参数，确保参数可以从GPU复制到CPU并克隆
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return  # 返回处理后的参数字典


local_rank = None  # 初始化local_rank变量为None，用于记录当前进程的本地排名


def rank0_print(*args):
    # 只有当local_rank为0时才打印消息，用于分布式训练中的主进程打印
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # 检查是否启用了DeepSpeed ZeRO-3模式
    if deepspeed.is_deepspeed_zero3_enabled():
        # 如果启用，使用_zero3_consolidated_16bit_state_dict方法收集状态字典
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        # 如果未启用ZeRO-3模式
        if trainer.args.use_lora:
            # 如果使用LoRA，调用get_peft_state_maybe_zero_3函数获取状态字典
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            # 如果不使用LoRA，直接获取模型的状态字典
            state_dict = trainer.model.state_dict()
    # 如果应该保存模型且当前进程是主进程
    if trainer.args.should_save and trainer.args.local_rank == 0:
        # 调用_trainer._save方法保存模型到指定目录
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
        sources,  # 原始对话数据列表
        tokenizer: transformers.PreTrainedTokenizer,  # 使用的分词器
        max_len: int,  # 输入的最大长度
        system_message: str = "You are a helpful assistant."  # 系统消息的默认文本
) -> Dict:  # 返回一个字典，包含输入ID、标签和注意力掩码
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

    # 初始化输入ID和目标ID的列表
    input_ids, targets = [], []

    # 遍历每个对话
    for i, source in enumerate(sources):
        # 如果第一个元素不是用户，跳过它
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        # 初始化当前对话的输入ID和目标ID
        input_id, target = [], []
        # 构建系统消息的输入ID和目标ID
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
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
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        # 确保输入ID和目标ID的长度相同
        assert len(input_id) == len(target)
        # 如果输入ID长度不足max_len，用填充符填充
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        # 将处理后的输入ID和目标ID添加到列表中
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    # 将输入ID和目标ID转换为张量
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    # 返回包含输入ID、标签和注意力掩码的字典
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # 注意力掩码，非填充符的位置为1
    )


class SupervisedDataset(Dataset):
    """用于监督式微调的数据集类。"""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        # 打印格式化输入的消息
        rank0_print("Formatting inputs...")
        # 从原始数据中提取对话数据
        sources = [example["conversations"] for example in raw_data]
        # 使用预处理函数处理对话数据
        data_dict = preprocess(sources, tokenizer, max_len)

        # 将处理后的数据赋值给类的属性
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """用于监督式微调的懒加载数据集类。"""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 在懒加载模式下跳过预处理步骤
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        # 否则，对第i个样本进行预处理
        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        # 将预处理后的数据转换为字典格式
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        # 将预处理后的数据缓存起来
        self.cached_data_dict[i] = ret

        # 返回第i个样本的输入ID、标签和注意力掩码
        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:


    # 根据是否懒加载预处理数据来选择数据集类
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )

    # 打印加载数据的消息
    rank0_print("Loading data...")

    # 加载训练数据
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    # 如果提供了评估数据路径，则加载评估数据
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    # 返回包含训练数据集和评估数据集的字典
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank  # 声明全局变量 `local_rank` 用于分布式训练中的设备索引

    parser = transformers.HfArgumentParser(  # 创建一个参数解析器，用于从命令行接收参数
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)  # 定义需要解析的参数类型
    )
    (
        model_args,  # 模型相关的参数
        data_args,  # 数据处理相关的参数
        training_args,  # 训练过程中的配置参数
        lora_args,  # LoRA 相关的参数
    ) = parser.parse_args_into_dataclasses()  # 解析命令行参数并转换成相应的数据类实例

    # 如果启用了 DeepSpeed 并且是单 GPU 模式，则设置分布式类型为 DeepSpeed
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank  # 获取本地设备的 rank，用于多 GPU 分布式训练

    device_map = None  # 初始化设备映射，用于指定模型加载到哪个设备上
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获取世界大小，即参与训练的 GPU 总数
    ddp = world_size != 1  # 判断是否使用分布式数据并行（DDP）
    if lora_args.q_lora:  # 如果启用了 QLoRA（量化 LoRA）
        # 设置设备映射，如果是 DDP 模式则根据 LOCAL_RANK 设置设备，否则自动分配
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():  # 如果启用了 FSDP 或者 ZeRO3
            logging.warning("FSDP or ZeRO3 are incompatible with QLoRA.")  # 警告用户 FSDP 或 ZeRO3 与 QLoRA 不兼容

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()  # 判断是否为聊天模型

    # 如果启用了 LoRA 且不是 QLoRA 模式，并且启用了 ZeRO3 优化，且不是聊天模型，则抛出运行时错误
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {  # 设置模型加载时的关键字参数
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),  # 如果没有启用 ZeRO3，则降低 CPU 内存使用
    }

    # 加载模型配置
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,  # 模型名称或路径
        cache_dir=training_args.cache_dir,  # 缓存目录
        trust_remote_code=True,  # 是否信任远程代码
    )
    config.use_cache = False  # 关闭缓存机制，防止在训练过程中出现内存溢出

    # 根据配置加载预训练模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,  # 模型名称或路径
        config=config,  # 模型配置
        cache_dir=training_args.cache_dir,  # 缓存目录
        device_map=device_map,  # 设备映射
        trust_remote_code=True,  # 是否信任远程代码
        quantization_config=GPTQConfig(  # 如果启用了 QLoRA，则设置量化配置
            bits=4,  # 量化的位数
            disable_exllama=True  # 禁用 exllama 优化
        ) if training_args.use_lora and lora_args.q_lora else None,  # 如果未启用 QLoRA，则不设置量化配置
        **model_load_kwargs,  # 其他模型加载参数
    )

    # 加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,  # 模型名称或路径
        cache_dir=training_args.cache_dir,  # 缓存目录
        model_max_length=training_args.model_max_length,  # 模型最大长度
        padding_side="right",  # 填充方向
        use_fast=False,  # 是否使用快速版本的分词器
        trust_remote_code=True,  # 是否信任远程代码
    )
    tokenizer.pad_token_id = tokenizer.eod_id  # 设置填充标记为结束标记

    if training_args.use_lora:  # 如果启用了 LoRA
        if lora_args.q_lora or is_chat_model:  # 如果启用了 QLoRA 或者是聊天模型
            modules_to_save = None  # 不保存特定模块
        else:
            modules_to_save = ["wte", "lm_head"]  # 否则，保存特定模块

        # 配置 LoRA 参数
        lora_config = LoraConfig(
            r=lora_args.lora_r,  # LoRA 秩大小
            lora_alpha=lora_args.lora_alpha,  # LoRA 学习率缩放因子
            target_modules=lora_args.lora_target_modules,  # 目标模块列表
            lora_dropout=lora_args.lora_dropout,  # LoRA dropout 概率
            bias=lora_args.lora_bias,  # 是否调整偏置
            task_type="CAUSAL_LM",  # 任务类型
            modules_to_save=modules_to_save  # 需要保存的模块
        )

        if lora_args.q_lora:  # 如果启用了 QLoRA
            model = prepare_model_for_kbit_training(  # 准备模型进行 k-bit 训练
                model, use_gradient_checkpointing=training_args.gradient_checkpointing  # 是否使用梯度检查点
            )

        model = get_peft_model(model, lora_config)  # 将 LoRA 应用到模型上

        # 打印可训练参数的数量
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:  # 如果启用了梯度检查点
            model.enable_input_require_grads()  # 启用输入梯度计算

    # 加载数据集
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,  # 分词器
        data_args=data_args,  # 数据参数
        max_len=training_args.model_max_length  # 最大长度
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,  # 模型
        tokenizer=tokenizer,  # 分词器
        args=training_args,  # 训练参数
        **data_module  # 数据集模块
    )

    trainer.train()  # 开始训练
    trainer.save_state()  # 保存训练状态


if __name__ == "__main__":
    train()
