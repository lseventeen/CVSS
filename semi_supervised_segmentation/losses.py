from torch import nn, Tensor
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch
from torch.autograd import Function, Variable
def softmax_helper(x): return F.softmax(x, 1)

class pCELoss(nn.Module):
    def __init__(self, reduction='none', pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(pCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pos_weight)

    def forward(self, prediction, target):
        mask = target == 255
        loss = self.bce_loss(prediction, target)
        if mask.sum() > 0:
            loss *= (~mask).float()
            loss = loss.sum()/(~mask).float().sum()
        else:
            loss = loss.mean()
        return loss

class pDiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(pDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        mask = target == 255
        prediction = torch.sigmoid(prediction)
        if mask.sum() > 0:
            prediction = prediction*(~mask).float()
            target = target*(~mask).float()

        
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss
def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent
class Dice_Loss(nn.Module):
    def __init__(self, n_classes=2, weight=None, softmax=True,ignore_index=255):
        super(Dice_Loss, self).__init__()
        self.n_classes = n_classes
        self.softmax = softmax
        self.ignore_index = ignore_index
        if weight is None:
            self.weight = [1] * self.n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.stack(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        
        # assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i],ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weight[i]
        return loss / self.n_classes








class CE_DiceLoss(nn.Module):
    def __init__(self, D_weight=0.5,ignore_index=255):
        super(CE_DiceLoss, self).__init__()
        self.DiceLoss = Dice_Loss(ignore_index=ignore_index)
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        return self.D_weight * self.DiceLoss(prediction, targets) + (1 - self.D_weight) * self.CELoss(prediction,
                                                                                                       targets)


