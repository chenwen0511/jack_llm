# 导入必备工具包
import torch
import numpy as np
from rich import print
from datasets import load_dataset
from transformers import AutoTokenizer
# import sys
# sys.path.append('..')
from ptune_config import *
from functools import partial


def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        p_embedding_num=6,
        train_mode=True,
        return_tensor=False
) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '娱乐	嗨放派怎么停播了',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        p_embedding_num (int): p-tuning token 的个数
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'mask_positions': [[5, 6, ...], [3, 4, ...]],
                            'mask_labels': [[183, 234], [298, 322], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'attention_mask': [],
        'mask_positions': [],  # 记录label的位置（即MASK Token的位置）
        'mask_labels': []  # 记录MASK Token的原始值（即Label值）
    }
    # print(f'examples--》{examples}')
    for i, example in enumerate(examples['text']):
        try:
            start_mask_position = 1  # 将 prompt token(s) 插在 [CLS] 之后

            if train_mode:
                label, content = example.strip().split('\t', 1)
            else:
                content = example.strip()

            encoded_inputs = tokenizer(
                text=content,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            continue

        input_ids = encoded_inputs['input_ids']
        # print(f'encoded_inputs-->{encoded_inputs}')
        # print(f'input_ids-->{input_ids}')

        mask_tokens = ['[MASK]'] * max_label_len  # 1.生成 MASK Tokens, 和label长度一致
        # print(f'mask_tokens-->{mask_tokens}')
        mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)  # token 转 id
        # print(f'mask_ids-->{mask_ids}')
        p_tokens = ["[unused{}]".format(i + 1) for i in range(p_embedding_num)]  # 2.构建 prompt token(s)
        # print(f'p_tokens-->{p_tokens}')

        p_tokens_ids = tokenizer.convert_tokens_to_ids(p_tokens)  # token 转 id
        # print(f'p_tokens_ids-->{p_tokens_ids}')
        tmp_input_ids = input_ids[:-1]
        # print(f'tmp_input_ids-->{len(tmp_input_ids)}')

        tmp_input_ids = tmp_input_ids[
                        :max_seq_len - len(mask_ids) - len(p_tokens_ids) - 1]  # 根据最大长度-p_token长度-label长度，裁剪content的长度
        # print(f'tmp_input_ids1-->{tmp_input_ids}')
        # print(len(tmp_input_ids))
        tmp_input_ids = tmp_input_ids[:start_mask_position] + mask_ids + tmp_input_ids[
                                                                         # 3.插入 MASK -> [CLS][MASK][MASK]世界杯...[SEP]
                                                                         start_mask_position:]
        # print(f'tmp_input_ids2--{tmp_input_ids}')

        input_ids = tmp_input_ids + [input_ids[-1]]  # 补上[SEP]
        # print(f'input_ids11-->{input_ids}')

        input_ids = p_tokens_ids + input_ids  # 4.插入 prompt -> [unused1][unused2]...[CLS][MASK]...[SEP]
        # print(f'input_ids22-->{input_ids}')
        # print(f'input_ids33-->{len(input_ids)}')
        mask_positions = [len(p_tokens_ids) + start_mask_position + i for  # 将 Mask Tokens 的位置记录下来
                          i in range(max_label_len)]

        tokenized_output['input_ids'].append(input_ids)
        # 如果输入需要token_type_ids，可以进行添加，

        if 'token_type_ids' in encoded_inputs:  # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
            tmp = encoded_inputs['token_type_ids']
            if 'token_type_ids' not in tokenized_output:
                tokenized_output['token_type_ids'] = [tmp]
            else:
                tokenized_output['token_type_ids'].append(tmp)
        tokenized_output['attention_mask'].append(encoded_inputs['attention_mask'])
        tokenized_output['mask_positions'].append(mask_positions)

        if train_mode:
            mask_labels = tokenizer(text=label)  # label token 转 id
            mask_labels = mask_labels['input_ids'][1:-1]  # 丢掉[CLS]和[SEP]
            mask_labels = mask_labels[:max_label_len]
            mask_labels += [tokenizer.pad_token_id] * (max_label_len - len(mask_labels))  # 将 label 补到最长
            tokenized_output['mask_labels'].append(mask_labels)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output





if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset('text', data_files={'train': pc.train_path})
    print(f'train_dataset==>{train_dataset}')
    # print(train_dataset['train']['text'])
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    #
    new_func = partial(convert_example,
                       tokenizer=tokenizer,
                       max_seq_len=20,
                       max_label_len=2,
                       p_embedding_num=6)

    dataset = train_dataset.map(new_func, batched=True)
    print(f'dataset---》{dataset}')
    a = dataset["train"][0]
    print(a)

