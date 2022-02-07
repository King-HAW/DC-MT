import torch
from torch.nn import functional as F


def cls_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.softmax(input_logits, dim=1)
    target_softmax = torch.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    bs = input_logits.size()[0]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / (num_classes * bs)


def att_mse_loss(mask, cams):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert mask.size() == cams.size() and len(mask.size()) == 4
    mse_loss = F.mse_loss(mask, cams, reduction='none').sum((2, 3))
    norm = (mask.sum((2, 3)) + cams.sum((2, 3))).sum()
    mse_loss = torch.sum(mse_loss) / torch.clamp(norm, min=1e-5)
    return mse_loss
