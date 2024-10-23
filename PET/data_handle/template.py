# -*- coding:utf-8 -*-
from rich import print
from transformers import AutoTokenizer
import numpy as np
import sys
import re
sys.path.append('..')
from pet_config import *

class HardTemplate(object):
    """
    硬模板，人工定义句子和[MASK]之间的位置关系。
    """

    def __init__(self, prompt: str):
        """
        Args:
            prompt (str): prompt格式定义字符串, e.g. -> "这是一条{MASK}评论：{textA}。"
        """
        self.prompt = prompt
        self.inputs_list = []                       # 根据文字prompt拆分为各part的列表
        self.custom_tokens = set()          # 从prompt中解析出的自定义token集合
        self.custom_tokens.add("MASK")
        self.prompt_analysis()                         # 解析prompt模板

    def prompt_analysis(self):
        """
        将prompt文字模板拆解为可映射的数据结构。

        Examples:
            prompt -> "这是一条{MASK}评论：{textA}。"
            inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
            custom_tokens -> {'textA', 'MASK'}
        """
        # 使用非贪婪模式来匹配花括号中的内容或普通字符
        pattern = r'(\{.*?\}|.)'
        # 使用findall方法来获取所有匹配的部分
        for word in re.findall(pattern, self.prompt):
            if word.startswith("{"):
                word = word[1:-1]
                self.custom_tokens.add(word)
            self.inputs_list.append(word)

    def __call__(self,
                 inputs_dict: dict,
                 tokenizer,
                 mask_length,
                 max_seq_len=512):
        """
        输入一个样本，转换为符合模板的格式。

        Args:
            inputs_dict (dict): prompt中的参数字典, e.g. -> {
                                                            "textA": "这个手机也太卡了", 
                                                            "MASK": "[MASK]"
                                                        }
            tokenizer: 用于encoding文本
            mask_length (int): MASK token 的长度

        Returns:
            dict -> {
                'text': '[CLS]这是一条[MASK]评论：这个手机也太卡了。[SEP]',
                'input_ids': [1, 47, 10, 7, 304, 3, 480, 279, 74, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2],
                'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'mask_position': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        """

        def custom_token_map(value):
            if value in self.custom_tokens:
                return inputs_dict[value] * mask_length if value == 'MASK' else inputs_dict[value]
            return value
        str_formated = "".join([custom_token_map(word) for word in self.inputs_list])
        print(f'str_formated-->{str_formated}')
        encoded = tokenizer(text=str_formated,
                            truncation=True,
                            max_length=max_seq_len,
                            padding='max_length')
        print(f'encoded--->{encoded}')
        outputs = dict()
        outputs['input_ids'] = encoded['input_ids']
        outputs['token_type_ids'] = encoded['token_type_ids']
        outputs['attention_mask'] = encoded['attention_mask']
        outputs['text'] = ''.join(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
        mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        mask_position = np.where(np.array(outputs['input_ids']) == mask_token_id)[0].tolist()
        outputs['mask_position'] = mask_position
        return outputs


if __name__ == '__main__':
    pc = ProjectConfig()
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')
    print(hard_template.inputs_list)
    print(hard_template.custom_tokens)
    tep = hard_template(
                inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
                tokenizer=tokenizer,
                max_seq_len=30,
                mask_length=2)
    print(tep)

    # print(tokenizer.convert_ids_to_tokens([3819, 3352, 3819, 3352]))
    # print(tokenizer.convert_tokens_to_ids(['网', '球']))
    """
['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA']
{'textA', 'MASK'}
str_formated-->这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。
encoded--->{'input_ids': [101, 6821, 3221, 671, 3340, 103, 103, 6397, 6389, 
8038, 1259, 6163, 679, 7231, 8024, 5741, 3362, 2923, 4494, 4638, 8024, 702, 
1928, 738, 1920, 511, 102, 0, 0, 0], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 0, 0, 0]}
{
    'input_ids': [
        101,
        6821,
        3221,
        671,
        3340,
        103,
        103,
        6397,
        6389,
        8038,
        1259,
        6163,
        679,
        7231,
        8024,
        5741,
        3362,
        2923,
        4494,
        4638,
        8024,
        702,
        1928,
        738,
        1920,
        511,
        102,
        0,
        0,
        0
    ],
    'token_type_ids': [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    'attention_mask': [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0
    ],
    'text': 
'[CLS]这是一条[MASK][MASK]评论：包装不错，苹果挺甜的，个头也大。[SEP][PAD][PAD]
[PAD]',
    'mask_position': [5, 6]
}    
    
    
    
    
    """
