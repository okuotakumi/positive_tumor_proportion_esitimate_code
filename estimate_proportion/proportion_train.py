from pathlib import Path
import argparse
import warnings
from tqdm import tqdm
warnings.simplefilter('ignore')
import torch
import matplotlib.pyplot as plt
from torch import optim

from estimate_proportion.utils import Train
from estimate_proportion.model import proportion_model
from estimate_proportion.model import proportion_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return tuple(zip(*batch))

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)



def train(model, dataloader, optimizer, criterion, save_path):
    model.train()
    loss_list = []
    early_counter = 0

    for epoch in tqdm(range(2000)):
        print(f"{epoch}/2000")
        losses = 0
        for imgs, img_paths, masks, gts, idxs in tqdm(dataloader):
            imgs = tuple_of_tensors_to_tensor(imgs)
            masks = tuple_of_tensors_to_tensor(masks)
            imgs = imgs.to(device)
            masks = masks.to(device)
            output,p_imgs,n_imgs,p_imgs_mask,n_imgs_mask = model(imgs, masks)
            
            gts = torch.tensor(gts)
            gt = gts.to(device)

            loss = criterion(output,gt)
            if torch.isnan(loss) == True:
                print(loss)
                print("break")
                break

            losses += float(loss)

            optimizer.zero_grad()
            loss.backward()  
            optimizer.step() 

        if torch.isnan(loss) == True:
            print("break")
            break
        epoch_loss = losses / (len(dataloader))
        print(f"loss{epoch}: {epoch_loss}")
        if loss_list == []:
            torch.save(model.state_dict(), str(save_path))
        else:
            if epoch_loss < min(loss_list):
                torch.save(model.state_dict(), str(save_path))
                early_counter = 0
            else:
                early_counter += 1

        loss_list.append(epoch_loss)
        if early_counter == 30:
            print("break")
            break

    plt.plot(range(len(loss_list)), loss_list, color = 'red')
    plt.savefig(str(save_path.parent.joinpath("loss_curve.png")))
    plt.close()



    


def main(args):
    batch_size = 4
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = proportion_model.Proportion_model(n_channels=3, n_classes=1)
    model = model.to(device)

    train_dataset = Train(args.base_path)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = proportion_loss.ProportionLoss_focal()
    train(model, data_loader, optimizer, criterion, save_path)


def parse_args():
    
    #Parse input arguments
    
    parser = argparse.ArgumentParser(description="estimate__proportion")
    parser.add_argument("--base_path", default="./estimate_proportion/datas/for_pred", type=str)
    parser.add_argument("--save_path", default="./estimate_proportion/weight/train/best.pth", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
