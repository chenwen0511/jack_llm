# coding:utf-8
from torch.utils.data import DataLoader
from transformers import default_data_collator
from data_preprocess import *
from pet_config import *

pc = ProjectConfig() # 实例化项目配置文件
tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)

def get_data():
    prompt = open(pc.prompt_file, 'r', encoding='utf8').readlines()[0].strip()  # prompt定义
    hard_template = HardTemplate(prompt=prompt)  # 模板转换器定义
    dataset = load_dataset('text', data_files={'train': pc.train_path,
                                               'dev': pc.dev_path})
    # print(dataset)
    # print(f'Prompt is -> {prompt}')
    new_func = partial(convert_example,
                       tokenizer=tokenizer,
                       hard_template=hard_template,
                       max_seq_len=pc.max_seq_len,
                       max_label_len=pc.max_label_len)

    dataset = dataset.map(new_func, batched=True)

    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    # print('train_dataset', train_dataset[:2])
    # print('*'*80)
    # default_data_collator作用，转换为tensor数据类型
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=default_data_collator,
                                  batch_size=pc.batch_size)
    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=default_data_collator,
                                batch_size=pc.batch_size)
    return train_dataloader, dev_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    print(len(train_dataloader))
    print(len(dev_dataloader))
    for i, value in enumerate(train_dataloader):
        print(i)
        print(value)
        print(value['input_ids'].dtype)
        break
