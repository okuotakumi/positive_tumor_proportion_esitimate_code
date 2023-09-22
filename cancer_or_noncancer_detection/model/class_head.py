from turtle import forward
import torch.nn as nn

class Class_head(nn.Module):
    def __init__(self, infeature, outfeature) -> None:
        super().__init__()
        self.linear = nn.Linear(infeature, outfeature)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.linear(x)
        out = self.sigmoid(out)
        out = out.squeeze()
        return out