from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import warnings
warnings.simplefilter('ignore')
from cancer_or_noncancer_detection.utils import Train
from cancer_or_noncancer_detection.model import c_or_n_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return tuple(zip(*batch))

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)



def train(model, dataloader, optimizer, criterion, save_path):
    model.train()
    loss_list = []
    for epoch in tqdm(range(100)):
        print(f"{epoch}/100")
        losses = 0
        for img, cpoints, npoints, point_bool in tqdm(dataloader):
            if sum(point_bool) == 0:
                continue
            img = tuple_of_tensors_to_tensor(img)
            input = img.to(device)

            output, c_len, length = model(input, cpoints, npoints) 
            
            #細胞単体が一つも取れなかったときに以下の処理を飛ばす
            if type(output) == list:
                continue
            if length == 0 | 1:
                continue

            gt = torch.zeros_like(output)
            gt[:c_len] = 1
            gt = gt.to(device)

            loss = criterion(output,gt)

            losses += loss.item()

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 

        epoch_loss = losses / (len(dataloader) + 1)
        print(f"loss{epoch}: {epoch_loss}")
        if loss_list == []:
            torch.save(model.state_dict(), str(save_path))
        else:
            if epoch_loss < min(loss_list):
                torch.save(model.state_dict(), str(save_path))
        loss_list.append(epoch_loss)
    plt.plot(range(100), loss_list)
    plt.savefig(str(save_path.parent.joinpath("loss_curve.png")))
    plt.close()


def main(args):
    batch_size = 16
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = c_or_n_model.Conv_crop(n_channels=3, n_classes=1)
    model = model.to(device)

    train_dataset = Train(args.img_path)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.BCELoss()
    train(model, data_loader, optimizer, criterion, save_path)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="cancer_or_nocancer_train")
    parser.add_argument("--img_path", default="./cancer_or_noncancer_detection/datas/train_data", type=str)
    parser.add_argument("--save_path", default="./cancer_or_noncancer_detection/weight/cancer_noncancer_detection/best.pth", type=str,)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)