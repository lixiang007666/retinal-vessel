import torch

def metrics(predict, target):
    # 统计预测信息
    # _, predicted = torch.max(predict.data, 1)
    total = target.sum().numpy()
    correct = predict[target == 1].squeeze().sum().numpy()
    accuracy = correct / total
    in_aero = correct
    dice = in_aero * 2 / (total + predict.sum().numpy())
    return accuracy, dice

import numpy as np
def hard_dice_coe(label, predict, num_class, smooth=1e-10):
    # indices = torch.max(predict, dim=1)[1]
    result = []
    for i in range(num_class):
        truth = (label == i).type(torch.float)
        pred = (predict == i).type(torch.float)
        inse = torch.sum(truth * pred, dim=[1, 2])
        l = torch.sum(pred * pred, dim=[1, 2])
        r = torch.sum(truth * truth, dim=[1, 2])
        re = 2 * inse / (l + r + smooth)
        result.append(re.cpu().numpy())
    result = np.stack(result, axis=-1)
    # rrr = np.mean(result, axis=0)
    return np.mean(result, axis=0)


import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        # loss = 1 - loss.sum() / N

        return loss.cpu().numpy()


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

# def acc(label, predict, num_class):
#     label_shape = label.shape
#     rank = len(label_shape)
#
#     ret = []
#     for i in range(num_class):
#         gt = tf.cast(tf.equal(label, i), tf.float32)
#         la = tf.cast(tf.equal(predict, i), tf.float32)
#         inse = tf.reduce_sum(gt*la, axis=list(range(1, rank)))
#         l = tf.reduce_sum(gt, axis=list(range(1, rank)))
#         acc = tf.reduce_mean((inse) / l)
#         # summary_ops.append(tf.summary.scalar('dice_{}'.format(i), dice))
#         ret.append(tf.reduce_mean(acc))
#
#     return ret