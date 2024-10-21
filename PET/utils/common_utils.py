# 导入必备工具包

import torch
from rich import print


def mlm_loss(logits, mask_positions, sub_mask_labels,
             cross_entropy_criterion, device):
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label, 由于每个label的sub_label数目不同，所以这里是个变长的list,
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        device (str): cpu还是gpu

    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    # print(f'mask_positions-->{mask_positions.shape}')
    # print(f'sub_mask_labels-->{sub_mask_labels}')
    loss = None
    # for single_logits, single_sub_mask_labels, single_mask_positions in zip(logits, sub_mask_labels, mask_positions):
    for single_value in zip(logits, sub_mask_labels, mask_positions):
        single_logits = single_value[0]
        # print(f'single_logits-->{single_logits.shape}')
        single_sub_mask_labels = single_value[1]
        # print(f'single_sub_mask_labels-->{single_sub_mask_labels}')
        single_mask_positions = single_value[2]
        # print(f'single_mask_positions-->{single_mask_positions}')
        # print(f'single_logits1-->{single_logits.shape}')
        # print(f'single_sub_mask_labels2-->{single_sub_mask_labels}')
        # print(f'single_mask_positions3-->{single_mask_positions}')
        #todo:single_logits-->形状[512, 21128],
        #todo:single_mask_positions--》形状size[2]-->具体值([5, 6])
        single_mask_logits = single_logits[single_mask_positions]  # (mask_label_num, vocab_size)
        # print(f'single_mask_logits4--<{single_mask_logits.shape}')
        # repeat重复的倍数
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1,
                                                       1)  # (sub_label_num, mask_label_num, vocab_size)
        # print(f'single_mask_logits5:{single_mask_logits.shape}')


        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)  # (sub_label_num * mask_label_num, vocab_size)
        # print(f'single_mask_logits6:{single_mask_logits.shape}')

        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)  # (sub_label_num, mask_label_num)
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()  # (sub_label_num * mask_label_num)
        # print(f'single_sub_mask_labels7-->{single_sub_mask_labels.shape}')

        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        cur_loss = cur_loss / len(single_sub_mask_labels)

        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss

    loss = loss / batch_size  # (1,)
    return loss


def convert_logits_to_ids(
        logits: torch.tensor,
        mask_positions: torch.tensor):
    """
    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)
    """
    label_length = mask_positions.size()[1]  # 标签长度
    # print(f'label_length--》{label_length}')
    batch_size, seq_len, vocab_size = logits.size()

    mask_positions_after_reshaped = []
    # print(f'mask_positions.detach().cpu().numpy().tolist()-->{mask_positions.detach().cpu().numpy().tolist()}')
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)
    # print(f'mask_positions_after_reshaped-->{mask_positions_after_reshaped}')
    # print(f'原始的logits-->{logits.shape}')
    logits = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
    # print('改变原始模型输出的结果形状', logits.shape)
    mask_logits = logits[mask_positions_after_reshaped]  # (batch * label_num, vocab_size)
    # print('选择真实掩码位置预测的数据形状',mask_logits.shape)
    predict_tokens = mask_logits.argmax(dim=-1)  # (batch * label_num)
    # print('求出每个样本真实mask位置预测的tokens', predict_tokens)
    predict_tokens = predict_tokens.reshape(-1, label_length)  # (batch, label_num)

    return predict_tokens


if __name__ == '__main__':
    logits = torch.randn(2, 20, 21193)
    mask_positions = torch.LongTensor([
        [5, 6],
        [5, 6],
    ])
    predict_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predict_tokens)
