import argparse

def sample_ica_args():   
    parser = argparse.ArgumentParser(description="ica")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--save_dir", default="./datas/core_data/colorchange", type=str)
    args = parser.parse_args()
    return args

def img_slice_args():
    parser = argparse.ArgumentParser(description="img_slice")
    parser.add_argument("--base_path", default="./datas/core_data/colorchange", type=str)
    parser.add_argument("--save_path", default="./base_detector/datas/patch_data", type=str,)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--slide", default=512, type=int)
    args = parser.parse_args()
    return args

def base_detector_train_args():
    parser = argparse.ArgumentParser(description="base_detect_train")
    parser.add_argument("--img_path", default="./base_detector/datas/detection_traindata", type=str)
    parser.add_argument("--save_path", default="./base_detector/weight/celldetector/best.pth", type=str,)
    args = parser.parse_args()
    return args

def base_detector_pred_args():
    parser = argparse.ArgumentParser(description="base_detector_pred")
    parser.add_argument("--img_path", default="./base_detector/datas/patch_data", type=str)
    parser.add_argument("--model_path", default="./base_detector/weight/for_pred/best.pth", type=str)
    parser.add_argument("--save_dir", default="./base_detector/output/for_pred", type=str,)
    args = parser.parse_args()
    return args

def patch2core_args():
    parser = argparse.ArgumentParser(description="patch2core")
    parser.add_argument("--pred_dir", default="./base_detector/output/for_pred", type=str)
    parser.add_argument("--save_dir", default="./base_detector/output/patch2core", type=str,)
    parser.add_argument("--origin_dir", default="./datas/core_data/original", type=str)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--slide", default=512, type=int)
    args = parser.parse_args()
    return args

def point2patch_args():
    parser = argparse.ArgumentParser(description="imgpoint_slice")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--point_path", default="./base_detector/output/patch2core/txtfile", type=str)
    parser.add_argument("--save_path", default="./datas/patch_data", type=str,)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--slide", default=512, type=int)
    args = parser.parse_args()
    return args

def c_or_n_train_args():
    parser = argparse.ArgumentParser(description="cancer_or_nocancer_train")
    parser.add_argument("--img_path", default="./cancer_or_noncancer_detection/datas/train_data", type=str)
    parser.add_argument("--save_path", default="./cancer_or_noncancer_detection/weight/cancer_noncancer_detection/best.pth", type=str,)
    args = parser.parse_args()
    return args

def c_or_n_pred_args():
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--base_path", default="./datas/patch_data", type=str)
    parser.add_argument("--origin_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--model_path", default="./cancer_or_noncancer_detection/weight/for_pred/best.pth", type=str)
    parser.add_argument("--save_dir", default="./cancer_or_noncancer_detection/output/for_pred", type=str)
    args = parser.parse_args()
    return args

def img_resize_args():
    parser = argparse.ArgumentParser(description="img_resize")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--save_path", default="./estimate_proportion/datas/for_pred/resize_data", type=str)
    parser.add_argument("--size", default=2048, type=int)
    args = parser.parse_args()
    return args

def make_mask_args():
    parser = argparse.ArgumentParser(description="make_mask")
    parser.add_argument("--img_path", default="./datas/core_data/original", type=str)
    parser.add_argument("--point_path", default="./cancer_or_noncancer_detection/output/for_pred", type=str)
    parser.add_argument("--save_path", default="./estimate_proportion/datas/for_pred/mask", type=str)
    parser.add_argument("--resize_scale", default=512, type=int)
    parser.add_argument("--radius", default=1, type=int)
    args = parser.parse_args()
    return args

def proportion_train_args():
    parser = argparse.ArgumentParser(description="estimate__proportion")
    parser.add_argument("--base_path", default="./estimate_proportion/datas/for_pred", type=str)
    parser.add_argument("--save_path", default="./estimate_proportion/weight/train/best.pth", type=str)
    args = parser.parse_args()
    return args

def proportion_test_args():
    parser = argparse.ArgumentParser(description="estimate__proportion")
    parser.add_argument("--base_path", default="./estimate_proportion/datas/for_pred", type=str)
    parser.add_argument("--model_path", default="./estimate_proportion/weight/for_pred/best.pth", type=str)
    parser.add_argument("--save_dir", default="./estimate_proportion/output/for_pred", type=str)
    args = parser.parse_args()
    return args