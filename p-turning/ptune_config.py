# coding:utf-8
import torch


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.pre_model = r'D:\01-大模型\大模型训练营--课件\大模型训练营--课件\第七章：新零售行业评价决策系统\01-code\预训练模型\bert-base-chinese'
        self.train_path = r'D:\01-大模型\大模型训练营--课件\大模型训练营--课件\第七章：新零售行业评价决策系统\01-code\P-Tuning\data\train.txt'
        self.dev_path = r'D:\01-大模型\大模型训练营--课件\大模型训练营--课件\第七章：新零售行业评价决策系统\01-code\P-Tuning\data\dev.txt'
        self.verbalizer = r'D:\01-大模型\大模型训练营--课件\大模型训练营--课件\第七章：新零售行业评价决策系统\01-code\P-Tuning\data\verbalizer.txt'
        self.max_seq_len = 512
        self.batch_size = 8
        self.learning_rate = 5e-5
        # 权重衰减系数
        self.weight_decay = 0
        # 学习率预热的系数
        self.warmup_ratio = 0.06
        self.p_embedding_num = 6
        self.max_label_len = 2
        self.epochs = 50
        self.logging_steps = 10
        self.valid_steps = 20
        self.save_dir = r'D:\01-大模型\大模型训练营--课件\大模型训练营--课件\第七章：新零售行业评价决策系统\01-code\P-Tuning\checkpoints'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.verbalizer)
