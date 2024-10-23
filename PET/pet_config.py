# coding:utf-8
import os.path

import torch
import sys
# print(sys.path)

BASE_DIR = os.path.dirname(__file__)


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # windows电脑/linux服务器
        self.pre_model = BASE_DIR + r'\bert-base-chinese'
        self.train_path = BASE_DIR + r'\data\train.txt'
        self.dev_path = BASE_DIR + r'\data\dev.txt'
        self.prompt_file = BASE_DIR + r'\data\prompt.txt'
        self.verbalizer = BASE_DIR + r'\data\verbalizer.txt'
        self.max_seq_len = 512
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.max_label_len = 2
        self.epochs = 2
        self.logging_steps = 10
        self.valid_steps = 20
        self.save_dir = BASE_DIR + r'\checkpoints'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.prompt_file)
    print(pc.pre_model)
