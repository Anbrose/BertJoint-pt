import torch
import torch.nn as nn
import torch.nn.functional as F

class NQLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_start_logits, predict_end_logits, predict_answerType,
                start_position, end_position, answerType):

        def compute_loss(logits, labels, class_num):
            one_hot_labels = F.one_hot(labels, num_classes=class_num)
            probs = F.log_softmax(logits, -1)
            loss = -torch.mean(torch.sum(one_hot_labels * probs), -1)
            return loss

        position_classes = predict_start_logits.shape[1]
        start_loss = compute_loss(predict_start_logits, start_position, position_classes)
        end_loss = compute_loss(predict_end_logits, end_position, position_classes)
        answerType_loss = compute_loss(predict_answerType,answerType,5)

        total_loss = (start_loss + end_loss + answerType_loss) / 3.0
        return total_loss
