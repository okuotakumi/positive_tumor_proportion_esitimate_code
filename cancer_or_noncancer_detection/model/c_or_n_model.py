import torch.nn as nn
import torch.nn.functional as F
from cancer_or_noncancer_detection.model import crop_fucntion
from cancer_or_noncancer_detection.model import feature_extractor
from cancer_or_noncancer_detection.model import class_head

acitivation = {'relu':nn.ReLU(inplace=True),
               'leakly':nn.LeakyReLU(inplace=True)}

class Conv_crop(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True,task='detection'):
        super(Conv_crop, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.task = task

        self.scale_factor = 16
        self.mode = 'bilinear'
        self.align_corners = True

        from torchvision.ops import misc as misc_nn_ops

        self.feature_extractor = feature_extractor.__dict__['resnet50'](
            pretrained=False,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.crop_function = crop_fucntion.Crop_function()
        self.class_head = class_head.Class_head(1024,1)

    def forward(self, imgs, cpoints, npoints):
        imgs = self.feature_extractor(imgs)
        imgs = F.interpolate(input = imgs, scale_factor = self.scale_factor, mode = self.mode, align_corners = self.align_corners)
        batch_crop_imgs, c_len, length = self.crop_function(imgs, cpoints, npoints)
        feature = self.class_head(batch_crop_imgs)
        return feature, c_len, length


class Core_conv_crop(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True,task='detection'):
        super(Core_conv_crop, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.task = task

        self.scale_factor = 16
        self.mode = 'bilinear'
        self.align_corners = True

        from torchvision.ops import misc as misc_nn_ops
        # import feature_extractor

        self.feature_extractor = feature_extractor.__dict__['resnet50'](
            pretrained=False,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.crop_function = crop_fucntion.Core_crop_function()
        self.class_head = class_head.Class_head(1024,1)

    def forward(self, imgs, point):
        imgs = self.feature_extractor(imgs)
        imgs = F.interpolate(input = imgs, scale_factor = self.scale_factor, mode = self.mode, align_corners = self.align_corners)
        batch_crop_imgs, length = self.crop_function(imgs, point)
        if type(batch_crop_imgs) != list:
            batch_crop_imgs = self.class_head(batch_crop_imgs)
        return batch_crop_imgs, length

# 以下テスト用
# if __name__=='__main__':
#     import torchvision
#     from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#     # load a model pre-trained pre-trained on COCO
#     # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
#     imgs = torch.randn(8, 3, 256, 256)

#     import random
#     batch_cpoints = []
#     for i in range(8):
#         k = random.randint(0,0)
#         cpoints = []
#         for l in range(k):
#             cpoint = random.sample(range(255),2)
#             cpoints.append(cpoint)
#         cpoints_tensor = torch.as_tensor(cpoints, dtype=torch.float32)
#         batch_cpoints.append(cpoints_tensor)
#     batch_cpoints = tuple(batch_cpoints)

#     batch_npoints = []
#     for i in range(8):
#         k = random.randint(0,0)
#         npoints = []
#         for l in range(k):
#             npoint = random.sample(range(255),2)
#             npoints.append(npoint)
#         npoints_tensor = torch.as_tensor(npoints, dtype=torch.float32)
#         batch_npoints.append(npoints_tensor)
#     batch_npoints = tuple(batch_npoints)

#     model = conv_crop(n_channels=3)

#     y = model(imgs, batch_cpoints, batch_npoints)