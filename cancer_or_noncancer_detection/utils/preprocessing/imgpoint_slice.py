from pathlib import Path
import argparse

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 1000000000




def make_point_patch(point, top, bottom, right, left):
    point = point[(left < point[:,0])&(right > point[:,0])&(top < point[:,1])&(bottom > point[:,1])]
    point_annotation = np.zeros((len(point),1))
    # point_annotation = np.reshape(point_annotation,[point_annotation.shape[0],1])
    point = point[:,0:2]
    if point.shape != (0,):
        point_patch = np.empty_like(point)
        if point.shape == (2,):   
            point_patch[0] = point[0] - left
            point_patch[1] = point[1] - top
        else:
            point_patch[:,0] = point[:,0] - left
            point_patch[:,1] = point[:,1] - top
        point_core_patch = np.concatenate([point, point_patch], 1)
        point_core_patch = np.concatenate([point_core_patch,point_annotation],1)
    return point_core_patch



def make_patch(img_path, img, point, save_path, p_size, slide):
    height_slice = len(img) // p_size + 1
    width_slice = len(img[0]) // p_size + 1
    for i in range(height_slice):
        for j in range(width_slice):
            top = slide*i
            left = slide*j
            if i == height_slice - 1:
                bottom = len(img)
            else:
                bottom = top + p_size
            if j == width_slice - 1:
                right = len(img[0])
            else:
                right = left + p_size
            img_patch = img[top:bottom, left:right]
            
            point_core_patch = make_point_patch(point, top, bottom, right, left)
            cv2.imwrite(f"{save_path}/img/{img_path.name[:-4]}_{i}_{j}.png",img_patch)
            np.savetxt(f"{save_path}/point/{img_path.name[:-4]}_{i}_{j}.txt", point_core_patch, fmt='%05d')
     


def main(args):
    img_paths = sorted(Path(f"{args.base_path}").glob("*.png"))
   
    for img_path in tqdm(img_paths):
        save_path = Path(f"{args.save_path}/{img_path.name[:-4]}")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("point").mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(img_path))
        # img_plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img_path.name[:-11] =='96_1_PDL122C3':
            point = np.loadtxt(f"{args.point_path}/96_1_PDL122C3/txtfile/{img_path.name[-10:-7]}.txt")
        elif img_path.name[:-11] =='96_2_PDL122C3':
            point = np.loadtxt(f"{args.point_path}/96_2_PDL122C3/txtfile/{img_path.name[-10:-7]}.txt")
        elif img_path.name[:-11] =='96_3_PDL122C3':
            point = np.loadtxt(f"{args.point_path}/96_3_PDL122C3/txtfile/{img_path.name[-10:-7]}.txt")
        elif img_path.name[:-11] =='96_4_PDL122C3':
            point = np.loadtxt(f"{args.point_path}/96_4_PDL122C3/txtfile/{img_path.name[-10:-7]}.txt")
        else:
            point = np.loadtxt(f"{args.point_path}/other_96/txtfile/{img_path.name[:-4]}.txt")
        point = point[:,0:2]
        point = point[(point[:,0] < len(img[0]))&(point[:,1] < len(img))]
        make_patch(img_path, img, point,  save_path, args.patch_size, args.slide)

        

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="img_slice")
    parser.add_argument("--base_path", default="./datas/core_data/colorchange", type=str)
    parser.add_argument("--point_path", default="./outputs/new_test/base_pred/core_size", type=str)
    parser.add_argument("--save_path", default="./outputs/new_test/cancer_cell_detect/other_96_512", type=str,)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--slide", default=512, type=int)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)