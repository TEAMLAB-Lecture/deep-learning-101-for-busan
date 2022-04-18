import torch
import torch.nn as nn
import torch.nn.functional as F


class MyBCELoss(nn.Module):

    def __init__(self):
        super(MyBCELoss, self).__init__()
        return

    def forward(self, y_pred, y_true):
        # losses = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        # cost = losses.mean()  # loss mean
        loss = F.binary_cross_entropy(y_pred, y_true)
        # 위 처럼 수식을 직접 입력하는 것 대신에, PyTorch에서
        # 제공하는 cross entropy 함수로 쉽게 모델을 정의할 수도 있다!
        return loss



class L2Regularizer(nn.Module):

    """
    Documentation: description, input, output
    """

    def __init__(self, lambd):
        super(L2Regularizer, self).__init__()
        self.lambd = lambd
        return

    def forward(self, model):
        loss = 0
        for name, p in model.named_parameters():
            if "bias" not in name:
                loss = loss + torch.sum(p*p)
        loss = self.lambd*loss
        return loss
