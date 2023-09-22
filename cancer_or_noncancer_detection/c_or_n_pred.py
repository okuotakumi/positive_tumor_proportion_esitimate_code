from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import torch
from cancer_or_noncancer_detection.utils import Core_conv_crop_pred

from cancer_or_noncancer_detection.model import c_or_n_model
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def points_plot(img, cpoint, npoint, img_save_path, img_path, size):
    plt.imshow(img)
    if cpoint.shape != (0,4):
        plt.scatter(cpoint[:,0],cpoint[:,1],c='purple', s = size ,marker="x")
    #通常細胞もプロットしたい場合
    # if npoint.shape != (0,4):
    #     plt.scatter(npoint[:,0],npoint[:,1],c='blue', s = size , marker="x")
    plt.axis("off")
    if img.size > 3145728:
        plt.savefig(f"{img_save_path}/{img_path.name}.png",bbox_inches='tight', pad_inches=0, transparent=True,dpi = len(img)/640*100)
    else:
        plt.savefig(f"{img_save_path}/img/{img_path.name[:-4]}.png",bbox_inches='tight', pad_inches=0, transparent=True,dpi = 300)   
    plt.close()

def main(args):
    model = c_or_n_model.Core_conv_crop(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location="cuda:0"))
    model = model.to(device)
    base_paths = Path(args.base_path).iterdir()
    for base_path in tqdm(base_paths):
        save_path = Path(f"{args.save_dir}/{base_path.name}")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
        save_path.joinpath("point").mkdir(parents=True, exist_ok=True)
        test_dataset = Core_conv_crop_pred(base_path)
        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,drop_last=False) 

        model.eval()

        pred_img_path = Path(f"{args.origin_path}/{base_path.name}.tif")
        pred_img = cv2.imread(str(pred_img_path))
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        
        core_all_point = np.empty((0,3))
        for img, point, img_path, point_bool in data_loader:
            img_path = Path(img_path[0])
            if point_bool == 0:
                continue
            input = img.to(device) 
            if point.shape == torch.Size([1,4]):
                patch_point = point[:,2:4]
                core_point = point[:,0:2]
            else:
                patch_point = point[:,:,2:4]
                core_point = point[:,:,0:2]

            output, length = model(input, patch_point)

            output[(output > 0.5)] = 1
            output[(output <= 0.5)] = 0
            output = output.to('cpu').detach().numpy().copy()
            
            if output.shape == ():
                output = output.reshape([output.size,1])
            else:
                output = output.reshape([len(output),1])

            if patch_point.shape != torch.Size([1,2]):
                patch_point = patch_point[0]
                core_point = core_point[0]
            patch_point = np.concatenate([patch_point, output], 1)
            core_point = np.concatenate([core_point, output], 1)

            

            core_all_point = np.append(core_all_point, core_point, axis = 0)

            #パッチ画像にプロットしたい場合
            # img = img[0] 
            # img = img.permute(1,2,0)
            # img = img.to('cpu').detach().numpy()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # patch_cpoint = patch_point[patch_point[:,2] == 1] 
            # patch_npoint = patch_point[patch_point[:,2] == 0]
            # points_plot(img, patch_cpoint, patch_npoint, save_path, img_path, 5) 

            np.savetxt(f"{save_path}/point/{img_path.name[:-4]}.txt", patch_point, fmt='%05d')
        core_cpoint = core_all_point[core_all_point[:,2] == 1]
        core_npoint = core_all_point[core_all_point[:,2] == 0]

        points_plot(pred_img, core_cpoint, core_npoint, save_path, base_path, 0.01)
        np.savetxt(f"{save_path}/point.txt", core_all_point, fmt='%05d')

          





def parse_args():
    
    #Parse input arguments
    
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--base_path", default="./datas/patch_data", type=str)
    parser.add_argument("--origin_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--model_path", default="./cancer_or_noncancer_detection/weight/for_pred/best.pth", type=str)
    parser.add_argument("--save_dir", default="./cancer_or_noncancer_detection/output/for_pred", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
