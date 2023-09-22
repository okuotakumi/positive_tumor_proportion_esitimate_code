import torch
import torch.nn as nn

class Crop_function(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, batch_cpoints, batch_npoints):
        batch_crop_cimgs = []
        batch_crop_nimgs = []
        c_len = 0
        n_len = 0
        for idx, img in enumerate(imgs):
            cpoints = batch_cpoints[idx]
            npoints = batch_npoints[idx]
            batch_crop_cimgs, c_len = self.create_crop(img, cpoints, batch_crop_cimgs, c_len)
            batch_crop_nimgs, n_len = self.create_crop(img, npoints, batch_crop_nimgs, n_len)
        batch_crop_imgs = batch_crop_cimgs + batch_crop_nimgs
        length = c_len + n_len
        if len(batch_crop_imgs) != 0:
            batch_crop_imgs = torch.cat(batch_crop_imgs).reshape(len(batch_crop_imgs), *batch_crop_imgs[0].shape)
        return batch_crop_imgs, c_len, length
    
    def create_crop(self,img,points,batch_crop_imgs,len):
        if points.shape != torch.Size([2]):
            for point in points:
                crop_img = self.point_crop(img,point)
                batch_crop_imgs.append(crop_img)
                len = len + 1
        else:
            crop_img = self.point_crop(img,points)
            batch_crop_imgs.append(crop_img)
            len = len + 1
        return batch_crop_imgs, len

    def point_crop(self, img, point):

        point = point.long()
        crop_img = img[:, point[1], point[0]]
        return crop_img


class Core_crop_function(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self):
        super().__init__()

    def forward(self, imgs, batch_points):
        batch_crop_imgs = []
        length = 0
        for idx, img in enumerate(imgs):
            points = batch_points[idx]
            batch_crop_imgs, length = self.create_crop(img, points, batch_crop_imgs, length)
        if len(batch_crop_imgs) != 0:
            batch_crop_imgs = torch.cat(batch_crop_imgs).reshape(len(batch_crop_imgs), *batch_crop_imgs[0].shape)
        return batch_crop_imgs, length
    
    def create_crop(self,img,points,batch_crop_imgs,len):
        if points.shape != torch.Size([2]):
            for point in points:
                crop_img = self.point_crop(img,point)
                batch_crop_imgs.append(crop_img)
                len = len + 1
        else:
            crop_img = self.point_crop(img,points)
            batch_crop_imgs.append(crop_img)
            len = len + 1
        return batch_crop_imgs, len

    def point_crop(self, img, point):
        point = point.long()
        crop_img = img[:, point[1], point[0]]
        return crop_img


# 以下テスト用
# if __name__=='__main__':
#     imgs = torch.randn(8, 512, 256, 256)
#     imgs[0, 0, 5, 10] = 255
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

#     model = crop_function()

#     batch_crop_imgs, c_len = model(imgs, batch_cpoints, batch_npoints)
#     print(batch_crop_imgs.shape)