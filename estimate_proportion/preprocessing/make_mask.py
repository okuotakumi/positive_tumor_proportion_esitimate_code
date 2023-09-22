from pathlib import Path
import argparse

import numpy as np
import cv2


def img_mask(point, r, resize_scale):
    mask = np.full((resize_scale,resize_scale),0, dtype=np.uint8)
    cpoints = point[(point[:,2] == 1)]
    if r == 1:
        cpoints = cpoints[(0 < cpoints[:,0])&(resize_scale > cpoints[:,0])&(0 < cpoints[:,1])&(resize_scale > cpoints[:,1])]
        for cpoint in cpoints:
            x,y = cpoint[:2]
            x = int(x)
            y = int(y)
            mask[y][x] = 255
    else:
        for cpoint in cpoints:
            x,y = cpoint[:2]
            x = int(x)
            y = int(y)
            cv2.circle(mask,center=(x,y),radius=r,color=255,thickness=-1)
    return mask

def point_resize(point, resize_scale, divide_img):
    width = divide_img.shape[1]
    height = divide_img.shape[0]
    point[:,0] = point[:,0]*resize_scale/width
    point[:,1] = point[:,1]*resize_scale/height
    divide_point = np.round(point)
    divide_point = divide_point[(0 < point[:,0])&(resize_scale > point[:,0])&(0 < point[:,1])&(resize_scale > point[:,1])]
    return divide_point

def main(args):
    img_paths = sorted(Path(f"{args.img_path}").glob("*.tif"))
    save_path = Path(f"{args.save_path}/{args.resize_scale}")
    save_path.mkdir(parents=True, exist_ok=True)
    for img_path in img_paths:
        point = np.loadtxt(f"{args.point_path}/{img_path.name[:-4]}/point.txt")
        img = cv2.imread(str(img_path))
        divide_point = point_resize(point, args.resize_scale, img)
        point_mask_img = img_mask(divide_point, args.radius, args.resize_scale)
        cv2.imwrite(f"{save_path}/{img_path.name[:-4]}.png", point_mask_img)
        
        

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="make_mask")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--point_path", default="./cancer_or_noncancer_detection/output/for_pred", type=str)
    parser.add_argument("--save_path", default="./estimate_proportion/datas/for_pred/mask", type=str)
    parser.add_argument("--resize_scale", default=512, type=int)
    parser.add_argument("--radius", default=1, type=int)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)