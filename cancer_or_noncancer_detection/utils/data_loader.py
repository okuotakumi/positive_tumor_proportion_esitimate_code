from pathlib import Path
from PIL import Image

import numpy as np
import torch

import cv2
from scipy.ndimage.interpolation import rotate

from natsort import natsorted

import matplotlib.pyplot as plt #試し用

 
        
class Train(object):
    def __init__(self,base_path):
        self.img_paths = sorted(Path(f"{base_path}/img").glob("*.png"))
        self.cpoint_paths = sorted(Path(f"{base_path}/cpoint").glob("*.txt"))
        self.npoint_paths = sorted(Path(f"{base_path}/npoint").glob("*.txt"))

        self.crop_size = (256, 256)

        self.cpoint_bool = 0
        self.npoint_bool = 0
    
    def __len__(self):
        return len(self.img_paths)
    
    #様々なサイズの入力に対応するためpadding,cropを行う
    def padding_img(self,img, cpoint, npoint):
        if (img.shape[:2] < np.array(self.crop_size)).any():
            if img.shape[0] < self.crop_size[0]:
                pad_height = self.crop_size[0] - img.shape[0]            
            else:
                pad_height = 0
            pad_top = -(-pad_height // 2)
            pad_bottom = pad_height // 2

            if img.shape[1] < self.crop_size[1]:
                pad_width = self.crop_size[1] - img.shape[1]
            else:
                pad_width = 0
            pad_left = -(-pad_width // 2)
            pad_right = pad_width // 2
        
            img = np.pad(img, ((pad_top,pad_bottom),(pad_left,pad_right),(0,0)))

            if cpoint.shape != (0,):
                if cpoint.shape == (2,):
                    cpoint[0] = cpoint[0] + pad_left
                    cpoint[1] = cpoint[1] + pad_top
                else:
                    cpoint[:,0] = cpoint[:,0] + pad_left
                    cpoint[:,1] = cpoint[:,1] + pad_top
            if npoint.shape != (0,):
                if npoint.shape == (2,):
                    npoint[0] = npoint[0] + pad_left
                    npoint[1] = npoint[1] + pad_top
                else:
                    npoint[:,0] = npoint[:,0] + pad_left
                    npoint[:,1] = npoint[:,1] + pad_top
        return img, cpoint, npoint

    def random_crop_param(self, shape):
        h, w = shape
        if(shape[0] > 256):
            top = np.random.randint(0, h - self.crop_size[0])
        else:
            top = 0
        if(shape[1] > 256):
            left = np.random.randint(0, w - self.crop_size[1])
        else:
            left = 0
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def point_crop(self, point, top, bottom, left, right):
        if point.shape != (0,):
            if point.shape == (2,):
                point = point[(left < point[0])&(right > point[0])&(top < point[1])&(bottom > point[1])]
            else:
                point = point[(left < point[:,0])&(right > point[:,0])&(top < point[:,1])&(bottom > point[:,1])]
            if point.shape == (0,2):
                point = point.reshape([0])
                point_bool = 0
            else:
                point[:,0] = point[:,0] - left
                point[:,1] = point[:,1] - top
                point_bool = 1
        else:
            point_bool = 0
        return point, point_bool

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = img / 255
        
        cpoint_path = self.cpoint_paths[idx]
        cpoint = np.loadtxt(cpoint_path)

        npoint_path = self.npoint_paths[idx]
        npoint = np.loadtxt(npoint_path)

        img, cpoint, npoint = self.padding_img(img, cpoint, npoint)

        top, bottom, left, right = self.random_crop_param(img.shape[:2])
        img = img[top:bottom, left:right]

        cpoint, cpoint_bool = self.point_crop(cpoint, top, bottom, left, right)
        npoint, npoint_bool = self.point_crop(npoint, top, bottom, left, right)
        
        img = torch.from_numpy(img.astype(np.float32))
        cpoint = torch.as_tensor(cpoint, dtype=torch.float32)
        npoint = torch.as_tensor(npoint, dtype=torch.float32)
        point_bool = cpoint_bool + npoint_bool
        return img.permute(2, 0, 1), cpoint, npoint, point_bool



class Crop_conv_pred(object):
    def __init__(self,base_path):
        self.img_paths = sorted(Path(f"{base_path}/img").glob("*.png"))
        self.point_paths = sorted(Path(f"{base_path}point").glob("*.txt"))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = img / 255
        
        point_path = self.point_paths[idx]
        point = np.loadtxt(point_path)

        if point.shape != (0,):
            point_bool = 1
        else:
            point_bool = 0
        
        img = torch.from_numpy(img.astype(np.float32))
        point = torch.as_tensor(point, dtype=torch.float32)
        return img.permute(2, 0, 1), point, point, img_path, point_bool

class Core_conv_crop_pred(object):
    def __init__(self,base_path):
        self.img_paths = natsorted(Path(f"{base_path}/img").glob("*.png"))
        self.point_paths = natsorted(Path(f"{base_path}/point").glob("*.txt"))
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = img / 255
        
        point_path = self.point_paths[idx]
        point = np.loadtxt(point_path)
        if point.shape != (0,):
            point_bool = 1
        else:
            point_bool = 0
        
        img = torch.from_numpy(img.astype(np.float32))
        point = torch.as_tensor(point, dtype=torch.float32)
        return img.permute(2, 0, 1), point, img_path, point_bool



class cell_conv(object):
    def __init__(self,base_path):
        self.img_paths0 = sorted(Path(f"{base_path}/cross1/img").glob("*.png"))
        self.point_paths0 = sorted(Path(f"{base_path}/cross1/0or1").glob("*.txt"))
        self.img_paths1 = sorted(Path(f"{base_path}/cross2/img").glob("*.png"))
        self.point_paths1 = sorted(Path(f"{base_path}/cross2/0or1").glob("*.txt"))

        self.img_paths =self.img_paths0 + self.img_paths1
        self.point_paths = self.point_paths0 + self.point_paths1
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = img / 255
        
        point_path = self.point_paths[idx]
        file = open(f"{point_path}", 'r')
        point = file.read()
        file.close()
        point = int(point)


        #ため使用
        # print(img_path)
        # print(point_path)
        
        # 試し用
        # plt.imshow(img)
        # plt.show()
        # plt.close()
        # ここまで
        
        img = torch.from_numpy(img.astype(np.float32))
        point = torch.as_tensor(point, dtype=torch.float32)
        return img.permute(2, 0, 1), point, img_path

class cell_conv_pred(object):
    def __init__(self,base_path):
        self.img_paths = natsorted(Path(f"{base_path}/img").glob("*.png"),key = lambda x:x.name)
        self.point_paths = natsorted(Path(f"{base_path}/0or1").glob("*.txt"),key = lambda x:x.name)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        img_path = str(img_path)
        img = cv2.imread(img_path)
        img = img / 255
        
        point_path = self.point_paths[idx]
        file = open(f"{point_path}", 'r')
        point = file.read()
        file.close()
        point = int(point)


        #ため使用
        # print(img_path)
        # print(point_path)
        
        # 試し用
        # plt.imshow(img)
        # plt.show()
        # plt.close()
        # ここまで
        
        img = torch.from_numpy(img.astype(np.float32))
        point = torch.as_tensor(point, dtype=torch.float32)
        return img.permute(2, 0, 1), point, img_path
