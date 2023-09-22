from pathlib import Path
import argparse
import cv2



def main(args):
    img_paths = sorted(Path(f"{args.img_path}").glob("*.tif"))
    save_path = Path(f"{args.save_path}/{args.size}")
    save_path.mkdir(parents=True, exist_ok=True)
    for img_path in img_paths:
        img = cv2.imread(f"{img_path}")
        resize_img = cv2.resize(img, dsize = (args.size, args.size))
        cv2.imwrite(f"{save_path}/{img_path.name[:-4]}.png", resize_img)
        
        

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="img_resize")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--save_path", default="./estimate_proportion/datas/for_pred/resize_data", type=str)
    parser.add_argument("--size", default=2048, type=int)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)