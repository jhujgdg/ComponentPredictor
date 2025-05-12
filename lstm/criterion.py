import torch
import torch.nn as nn


class BinaryCrossEntropyLoss:
    def __init__(self):
        # 使用 PyTorch 的 BCEWithLogitsLoss
        self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, predictions, targets):
        """
        计算二元交叉熵损失

        :param predictions: 模型的预测值，未经过 Sigmoid 的 logits
        :param targets: 真实标签，应该是 0 或 1
        :return: 计算得到的损失值
        """
        # 确保 targets 是 float 类型
        targets = targets.float()

        # 计算损失
        loss = self.loss_fn(predictions, targets)
        return loss


def calculate_accuracy(true_labels, predictions, threshold=0.4):
    probabilities = torch.sigmoid(predictions)
    predicted_labels = (probabilities > threshold).float()

    # Debugging: Print predictions and true labels

    correct_predictions = (predicted_labels == true_labels).float()
    accuracy = correct_predictions.sum() / correct_predictions.numel()

    TP = (predicted_labels * true_labels).sum(dim=0)
    FP = (predicted_labels * (1 - true_labels)).sum(dim=0)
    FN = ((1 - predicted_labels) * true_labels).sum(dim=0)

    # Valid masks
    valid_mask1 = (TP + FP) > 0
    valid_mask2 = (TP + FN) > 0

    precision = torch.zeros_like(TP)
    recall = torch.zeros_like(TP)

    precision[valid_mask1] = TP[valid_mask1] / (TP[valid_mask1] + FP[valid_mask1] + 1e-10)
    recall[valid_mask2] = TP[valid_mask2] / (TP[valid_mask2] + FN[valid_mask2] + 1e-10)

    f1_score = torch.zeros_like(precision)
    valid_mask_f1 = (precision + recall) > 0

    f1_score[valid_mask_f1] = 2 * (precision[valid_mask_f1] * recall[valid_mask_f1]) / (
                precision[valid_mask_f1] + recall[valid_mask_f1] + 1e-10)

    return accuracy.item(), precision[valid_mask1].mean().item(), recall[valid_mask2].mean().item(), f1_score[
        valid_mask_f1].mean().item()
