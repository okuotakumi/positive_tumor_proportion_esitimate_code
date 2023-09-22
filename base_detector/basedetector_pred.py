from pathlib import Path
import argparse
import cv2
from tqdm import tqdm
import torch
from base_detector.model import UNet
from base_detector.utils import DetectionTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pred(model, test_dataset, dataloader, save_dir):
    model.eval()
    for img, idxs in tqdm(dataloader):
        img = img.to(device)
        output  = model(img)  # データを流す
        for hm, idx in zip(output, idxs):
            img_path = test_dataset.img_paths[idx]
            hm_re = hm[0].detach().cpu().numpy() * 255
            hm_re = hm_re.astype('uint8')
            save_path = save_dir.joinpath(f"{img_path.parent.name}/{img_path.name}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), hm_re)


def main(args):
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location="cuda:0"))
    model = model.to(device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    #core画像から持ってきたい場合はこっち
    test_dataset = DetectionTest(args.img_path)

    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    
    pred(model, test_dataset, data_loader, save_dir)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="base_detector_pred")
    parser.add_argument("--img_path", default="./base_detector/datas/patch_data", type=str)
    parser.add_argument("--model_path", default="./base_detector/weight/for_pred/best.pth", type=str)
    parser.add_argument("--save_dir", default="./base_detector/output/for_pred", type=str,)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

