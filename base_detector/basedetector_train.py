from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
from base_detector.utils import DetLoaderPatch
from base_detector.model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, save_path):
    model.train()
    loss_list = []
    for epoch in range(30):
        print(f"{epoch}/30")
        losses = 0
        for img, mask in tqdm(dataloader):
            input = img.to(device)
            mask = mask.to(device) 

            output = model(input) 

            loss = criterion(output, mask)

            losses += loss.item()

            optimizer.zero_grad()

            loss.backward()  
            optimizer.step()  
        # 中間の結果を記録
        epoch_loss = losses / (len(dataloader) + 1)
        print(f"loss: {epoch_loss}")
        if loss_list == []:
            torch.save(model.state_dict(), str(save_path))
        else:
            if epoch_loss < min(loss_list):
                torch.save(model.state_dict(), str(save_path))
        loss_list.append(epoch_loss)
    plt.plot(range(epoch+1), loss_list)
    plt.savefig(str(save_path.parent.joinpath("loss_curve.png")))
    plt.close()


def main(args):
    batch_size = 8
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = UNet(n_channels=3, n_classes=1)
    model = model.to(device)

    train_dataset = DetLoaderPatch(args.img_path)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train(model, data_loader, optimizer, criterion, save_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="base_detect_train")
    parser.add_argument("--img_path", default="./base_detector/datas/detection_traindata", type=str)
    parser.add_argument("--save_path", default="./base_detector/weight/celldetector/best.pth", type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
            
    