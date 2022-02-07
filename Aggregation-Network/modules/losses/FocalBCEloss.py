import torch
import torch.nn as nn

# https://blog.csdn.net/Code_Mart/article/details/89736187


class FocalBCEloss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalBCEloss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, y_pred, y_true):
        epsilon = 1e-10
        y_pred = y_pred + epsilon
        pred = y_pred.view(-1, 1)
        target = y_true.view(-1, 1)

        pred = torch.cat((1 - pred, pred), dim=1)

        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        log_p = probs.log()

        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        batch_loss = - alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


