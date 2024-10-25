import os
import time
import copy
import argparse
from functools import partial
import peft
# autocast是PyTorch中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
# 该方法混合精度训练，如果在CPU环境中不起任何作用
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_scheduler,AutoModelForCausalLM
from utils.common_utils import *
from data_handle.data_loader import *
from glm_config import *

pc = ProjectConfig()

def evaluate_model(model, dev_dataloader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in dev_dataloader:
            if pc.use_lora:
                with autocast():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            loss_list.append(float(loss.cpu().detach()))
    model.train()
    return sum(loss_list) / len(loss_list)


def model2train():
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)

    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)

    if pc.use_ptuning:
        config.pre_seq_len = pc.pre_seq_len
        config.prefix_projection = pc.prefix_projection
    model = AutoModel.from_pretrained(pc.pre_model,
                                      config=config,
                                      trust_remote_code=True)

    #model.half()将模型数据类型从默认的float32精度转换为更低的float16精度，减少内存
    model = model.float()
    print(model)
    # 梯度检查点是一种优化技术，用于在反向传播过程中降低内存使用
    # 保存部分激活值，未保存的反向传播时重新计算
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # 不进行缓存，减少内存
    model.config.use_cache = False

    if pc.use_ptuning:
        model.transformer.prefix_encoder.float()
    print(f'model.lm_head-->{model.lm_head}')
    if pc.use_lora:
        model.lm_head = CastOutputToFloat(model.lm_head)
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False, # 推理时为True，比如绝定是否使用dropout
            r=pc.lora_rank, # 低秩矩阵维度
            lora_alpha=32, # 缩放系数
            lora_dropout=0.1,
        )
        model = peft.get_peft_model(model, peft_config)

    print(f'model2-->{model}')
    model = model.to(pc.device)
    print('模型训练参数', model.print_trainable_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": pc.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=pc.learning_rate)
    # model.to(pc.device)
    #
    train_dataloader, dev_dataloader = get_data()
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    #指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = pc.epochs * num_update_steps_per_epoch
    warm_steps = int(pc.warmup_ratio * max_train_steps) # 预热阶段的训练步数
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    #
    loss_list = []
    tic_train = time.time()
    global_step, best_eval_loss = 0, float('inf')
    for epoch in range(1, pc.epochs + 1):
        print("开始训练")
        for batch in train_dataloader:
            if pc.use_lora:
                # torch.cuda.amp.autocast是PyTorch中一种混合精度的技术（仅在GPU上训练时可使用）
                with autocast():
                    loss = model(
                        input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                        labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                    ).loss
            else:
                loss = model(
                    input_ids=batch['input_ids'].to(dtype=torch.long, device=pc.device),
                    labels=batch['labels'].to(dtype=torch.long, device=pc.device)
                ).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % pc.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print("global step %d ( %02.2f%% ) , epoch: %d, loss: %.5f, speed: %.2f step/s, ETA: %s"
                      % (
                          global_step,
                          global_step / max_train_steps * 100,
                          epoch,
                          loss_avg,
                          pc.logging_steps / time_diff,
                          second2time(int(max_train_steps - global_step) / (pc.logging_steps / time_diff))
                      ))
                tic_train = time.time()

            if global_step % pc.save_freq == 0:
                cur_save_dir = os.path.join(pc.save_dir, "model_%d" % global_step)
                save_model(model, cur_save_dir)
                tokenizer.save_pretrained(cur_save_dir)
                print(f'Model has saved at {cur_save_dir}.')

                eval_loss = evaluate_model(model, dev_dataloader)

                print("Evaluation Loss: %.5f" % (eval_loss))
                if eval_loss < best_eval_loss:
                    print(
                        f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}"
                    )
                    best_eval_loss = eval_loss
                    cur_save_dir = os.path.join(pc.save_dir, "model_best")
                    save_model(model, cur_save_dir)
                    tokenizer.save_pretrained(cur_save_dir)
                    print(f'Best model has saved at {cur_save_dir}.')
                tic_train = time.time()


if __name__ == '__main__':
    model2train()