import torch
import torch.nn as nn

class Mask_sum(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, imgs, masks):
        imgs_softmax = self.softmax(imgs)
        p_imgs = imgs_softmax[:,0]
        n_imgs = imgs_softmax[:,1]
        imgs_mask = imgs_softmax.masked_fill(masks == 0, 0)
        p_imgs_mask = imgs_mask[:,0]
        n_imgs_mask = imgs_mask[:,1]
        imgs_sum = torch.sum(imgs_mask, 2)
        imgs_sum = torch.sum(imgs_sum, 2)
        output = imgs_sum[:,0] / torch.sum(imgs_sum,1)
        return output,p_imgs,n_imgs,p_imgs_mask,n_imgs_mask