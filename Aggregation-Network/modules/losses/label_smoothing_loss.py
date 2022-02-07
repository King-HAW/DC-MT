import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    '''
    label smoothing loss
    '''

    def __init__(self, epsilon, beta):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.beta = beta


    def forward(self, inputs, targets, flags):
        B, C = inputs.size()

        ones_mask = torch.ones(targets.size()).cuda()
        zeros_mask = torch.zeros(targets.size()).cuda()

        epsilon_flags = 1 - flags
        epsilon_flags = epsilon_flags*self.epsilon

        beta_flags = flags
        ones_flags = torch.ones(beta_flags.size()).cuda()
        beta_flags = torch.where(beta_flags==0, self.beta*ones_flags, ones_flags)

        # loss = targets*torch.log(inputs) + (1-targets)*torch.log(1-inputs)

        loss = (1-epsilon_flags)*(targets*torch.log(inputs) + (1-targets)*torch.log(1-inputs)) + epsilon_flags*((1-targets)*torch.log(inputs) + targets*torch.log(1-inputs))

        loss = loss * beta_flags 
        loss = -loss.sum() / (B*C)
        

        return loss         
