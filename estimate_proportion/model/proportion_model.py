import torch.nn as nn
import torch.nn.functional as F
from estimate_proportion.model import mask_sum
from estimate_proportion.model import feature_extractor




class Proportion_model(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True,task='detection'):
        super(Proportion_model, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.task = task

        self.scale_factor = 0.25
        self.mode = 'nearest'

        from torchvision.ops import misc as misc_nn_ops

        self.feature_extractor = feature_extractor.__dict__['resnet18'](
            pretrained=True,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.finalconv = nn.Conv2d(512, 2, kernel_size=1)
        self.mask_sum = mask_sum.Mask_sum()



    def forward(self, img, mask):
        conv_img = self.feature_extractor(img)
        # mask = F.interpolate(input = mask, size = (len(conv_img[0][0]), len(conv_img[0][0][0])), mode = self.mode)
        finalconv_img = self.finalconv(conv_img)
        up_img = F.interpolate(input = finalconv_img, size = (len(mask[0][0]), len(mask[0][0][0])), mode = self.mode)
        mask = mask[:,0:2]
        output, p_imgs, n_imgs, p_imgs_mask, n_imgs_mask = self.mask_sum(up_img, mask)
        return output,p_imgs,n_imgs,p_imgs_mask,n_imgs_mask

