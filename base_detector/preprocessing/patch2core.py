import argparse
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter('ignore')

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from skimage import feature
import cv2
from tqdm import tqdm


def local_maxima(img, threshold=0.3, dist=20):
    assert len(img.shape) == 2
    data = np.zeros((0, 2))
    x = feature.peak_local_max(img, threshold_abs=threshold*255,min_distance=dist)    
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data

def hm2points(hm_paths, hw_info, patch_size, slide):
    whole_hm = np.zeros((round(hw_info[1] / patch_size) * patch_size + patch_size, round(hw_info[0] / patch_size) * patch_size + patch_size), dtype=np.uint8)
    for hm_path in hm_paths:
        hm_l = np.array(Image.open(hm_path))
        h_idx, w_idx = hm_path.stem.split("_")
        w_idx, h_idx = int(w_idx), int(h_idx)
        t, b, l, r = h_idx * slide, h_idx * slide + len(hm_l), w_idx * slide, w_idx * slide + len(hm_l[0])
        whole_hm[t:b, l:r] = np.maximum(whole_hm[t:b, l:r], hm_l)
    whole_hm[:hw_info[1], :hw_info[0]] 
    points = local_maxima(whole_hm)
    points = points.astype(np.int64)

    scores = []
    for x, y in points:
        score = np.sum(whole_hm[y - 9:y + 9, x - 9:x + 9])
        scores.append(score)
    scores = np.array(scores)
    points = np.concatenate([points, scores[:, None]], axis=1)
    return points, whole_hm


def main(args):
    pred_dirs = Path(args.pred_dir).iterdir()
    save_dir = Path(args.save_dir)
    origin_dir = Path(args.origin_dir)
    
    for pred_dir in tqdm(pred_dirs):
        origin_img = cv2.imread(str(origin_dir.joinpath(pred_dir.name + ".tif")))
        w,h = origin_img.shape[0:2]
        hw_info = [h,w]
        hm_paths = pred_dir.glob("*.png")
        patch2core = hm2points(hm_paths, hw_info, args.patch_size, args.slide)
        #座標データを保存
        point_dir = save_dir.joinpath("txtfile")
        point_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt((f"{point_dir}/{pred_dir.name}.txt"), patch2core[0]) 
        #画像を保存
        img = Image.fromarray(patch2core[1].astype(np.uint8))
        img_dir = save_dir.joinpath("img")
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(f"{img_dir}/{pred_dir.name}.png") 
        origin_img = Image.open(origin_dir.joinpath(pred_dir.name + ".tif"))
        origin_rescale = origin_img.resize(size = (origin_img.width // 4, origin_img.height // 4), resample=Image.LANCZOS)
        patch2core_rescale = patch2core[0]/4
        origin_img_dir = save_dir.joinpath("origin_plot_img")
        origin_img_dir.mkdir(parents=True, exist_ok=True)
        plt.imshow(origin_rescale)
        plt.scatter(patch2core_rescale[:,0],patch2core_rescale[:,1],c='red',s=0.05,marker="x")
        plt.axis("off")
        plt.savefig(f"{origin_img_dir}/{pred_dir.name}.png",bbox_inches='tight', pad_inches=0, transparent=True, dpi=origin_rescale.size[0]/640*100)
        plt.close()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="patch2core")
    parser.add_argument("--pred_dir", default="./base_detector/output/for_pred", type=str)
    parser.add_argument("--save_dir", default="./base_detector/output/patch2core", type=str,)
    parser.add_argument("--origin_dir", default="./datas/core_data/original", type=str)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--slide", default=512, type=int)

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    main(args)