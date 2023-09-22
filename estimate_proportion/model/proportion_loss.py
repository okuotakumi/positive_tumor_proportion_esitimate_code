import torch
import torch.nn as nn
    


class ProportionLoss_focal(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ProportionLoss_focal, self).__init__()

    def forward(self, inputs, targets):
        targets = targets.to(torch.float32)
        focal = torch.zeros_like(targets)
        focal[targets == 2] = 0
        focal[targets == 3] = 2
        focal[targets == 4] = 2
        focal[targets == 5] = 2
        focal[targets == 6] = 2
        targets[targets == 2] = 0
        targets[targets == 3] = 0.13
        targets[targets == 4] = 0.375
        targets[targets == 5] = 0.625
        targets[targets == 6] = 1
        
        loss_val = torch.add(torch.mul(targets,torch.log(inputs)), torch.mul((1-targets),torch.log(1-inputs)))
        cor_val = torch.add(torch.mul(targets,torch.log(targets + 1e-10)), torch.mul((1-targets),torch.log(1-targets + 1e-10)))
        loss = loss_val - cor_val
        losses = torch.abs(torch.mul(torch.pow(torch.abs(targets - inputs),focal), loss))
        loss_mean = torch.mean(losses)
        return loss_mean