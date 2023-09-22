from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path #ファイル一覧で読み込む
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import h5py
import argparse
from tqdm import tqdm

def main(args):
    with h5py.File('./sample_ica/ICAcomponents.mat', 'r') as f:
    # with h5py.File('ICAcomponents.mat', 'r') as f:
        mu = np.array(f["mu"])
        w = np.array(f["W"])
        t = np.array(f["T"])
        wi = np.array(f["WI"])
    ww = np.linalg.lstsq(t, w)[0]
    list = Path(args.img_path).glob("*.tif") 
    for image in tqdm(list):
        img = np.array(Image.open(image).convert('RGB')) 
        
        # W, Hをflatに
        img_flat = img.reshape(-1, 3)
        w, h, c = img.shape 

        # ica decomposition
        ica_zaxis = wi.T @ (img_flat - mu).transpose(1, 0)
        
        ica_img = ica_zaxis.transpose(1, 0).reshape(w, h, 3)
        ica_img = (ica_img - ica_img.min()) / (ica_img.max() - ica_img.min())
        ica_decomposit_img = ica_img[:, :, 0].reshape(img.shape[:2])

        # decomposition by using ICA basis (RGB)
        ica_img = (ww[:, [2]] @ ica_zaxis[[2], :]) + mu.transpose(1, 0)
        ica_img = ica_img.reshape(3, w, h).transpose(1, 2, 0) #reshapeは画像サイズに合わせて

        ica_img = Image.fromarray(ica_img.astype(np.uint8))
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ica_img.save(f"{save_dir}/" + image.name)



def parse_args():
    
    #Parse input arguments
    
    parser = argparse.ArgumentParser(description="ica")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--save_dir", default="./datas/core_data/colorchange", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)