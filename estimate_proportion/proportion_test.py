from pathlib import Path
import argparse
import warnings

warnings.simplefilter('ignore')
import torch

from estimate_proportion.utils import Pred
from estimate_proportion.model import proportion_model
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from statistics import mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return tuple(zip(*batch))

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)


def np2heatmap(img,save_path,img_path,idx,mask_or_not):
    if mask_or_not:
        img[img!=0] = img[img!=0] - 0.5
        img = img*2
    plt.figure(figsize=(512,512))
    sns.heatmap(img, xticklabels=False, yticklabels=False, square=True, cmap='bwr', cbar=False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(f"{save_path}/{Path(img_path[idx]).name}",dpi=1)
    plt.close()



def main(args):
    model = proportion_model.Proportion_model(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location="cuda:0"))
    model = model.to(device)

    test_dataset = Pred(args.base_path)

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2,drop_last=False, collate_fn=collate_fn)
    pred_gts = np.empty((0,2))

    model.eval()
    for img, img_paths, mask, gt, idxs in data_loader:

        img = tuple_of_tensors_to_tensor(img)
        input = img.to(device)
        mask = tuple_of_tensors_to_tensor(mask)
        mask = mask.to(device)
        output,p_imgs,n_imgs,p_imgs_mask,n_imgs_mask = model(input, mask)
        
        #中間出力を保存したい場合
        # p_imgs =p_imgs.to('cpu').detach().numpy().copy()
        # p_imgs_mask =p_imgs_mask.to('cpu').detach().numpy().copy()
        # save_dir_premask = Path(f"{args.save_dir}/premask")
        # save_dir_premask.mkdir(parents=True, exist_ok=True)
        # save_dir_premask = str(save_dir_premask)
        # save_dir_masked = Path(f"{args.save_dir}/masked")
        # save_dir_masked.mkdir(parents=True, exist_ok=True)
        # save_dir_masked = str(save_dir_masked)
        # for i, (p_img,p_img_mask) in enumerate(zip(p_imgs,p_imgs_mask)):
        #     np2heatmap(p_img,save_dir_premask,img_paths,i, False)
        #     np2heatmap(p_img_mask,save_dir_masked,img_paths,i, True)
        
        output = output.to('cpu').detach().numpy().copy()
        output = output.reshape([len(output),1])

        gt = np.array(gt)
        gt = gt.reshape([len(output),1])
        pred = np.zeros_like(output)
        for i in range(len(idxs)):
            pred[i] = output[i]
        pred[(0 <= pred)&(pred <= 0.01)] = 2
        pred[(0.01 < pred)&(pred <= 0.25)] = 3
        pred[(0.25 < pred)&(pred <= 0.50)] = 4
        pred[(0.50 < pred)&(pred <= 0.75)] = 5
        pred[(0.75 < pred)&(pred <= 1.00)] = 6

        pred_gt = np.concatenate([pred, gt], 1)
        pred_gts = np.append(pred_gts, pred_gt, axis = 0)


    precisions = []
    recalls = []
    f1s = []
    for i in range(2,7):
        pred_i = len(pred_gt[pred_gt[:,0] == i])
        if pred_i == 0:
            pred_i = 1
        gt_i = len(pred_gt[pred_gt[:,1] == i])
        if gt_i == 0:
            gt_i = 1
        tp = len(pred_gt[(pred_gt[:, 0] == i) & (pred_gt[:, 1] == i)])
        if tp == 0:
            tp = 1
        precision = tp / pred_i
        recall = tp / gt_i
        f1 = 2*precision*recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    mrecall = mean(recalls)
    mprecision = mean(precisions)
    mf1 = mean(f1s)
    print(f"mRecall: {mrecall}")
    print(f"mPrecision: {mprecision}")
    print(f"mF1: {mf1}")



def parse_args():
    
    #Parse input arguments
    
    parser = argparse.ArgumentParser(description="estimate__proportion")
    parser.add_argument("--base_path", default="./estimate_proportion/datas/for_pred", type=str)
    parser.add_argument("--model_path", default="./estimate_proportion/weight/train/best.pth", type=str)
    parser.add_argument("--save_dir", default="./estimate_proportion/output/for_pred", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
