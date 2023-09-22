import param

from sample_ica import sample_ica
from base_detector.preprocessing import img_slice
from base_detector import basedetector_train
from base_detector import basedetector_pred
from base_detector.preprocessing import patch2core
from cancer_or_noncancer_detection.preprocessing import point2patch
from cancer_or_noncancer_detection import c_or_n_train
from cancer_or_noncancer_detection import c_or_n_pred
from estimate_proportion.preprocessing import img_resize
from estimate_proportion.preprocessing import make_mask
from estimate_proportion import proportion_train
from estimate_proportion import proportion_test



if __name__=='__main__':
    print("======process start=====")
    print("======color_change start=====")
    #color_change
    args = param.sample_ica_args()  
    sample_ica.main(args)

    print("=====base_detector start=====")
    #img_slice
    args = param.img_slice_args()
    img_slice.main(args)

    #base_detector_train
    args = param.base_detector_train_args()
    basedetector_train.main(args)

    #base_detector_pred
    args = param.base_detector_pred_args()
    basedetector_pred.main(args)

    #patch2core
    args = param.patch2core_args()
    patch2core.main(args)

    print("=====cancer_or noncancer_detection start=====")
    #point2patch
    args = param.point2patch_args()
    point2patch.main(args)

    #c_or_n_train
    args = param.c_or_n_train_args()
    c_or_n_train.main(args)

    #c_or_n_pred
    args = param.c_or_n_pred_args()
    c_or_n_pred.main(args)
    
    print("=====estimate proportion start=====")
    #img_resize
    args = param.img_resize_args()
    img_resize.main(args)

    #make_mask
    args = param.make_mask_args()
    make_mask.main(args)

    #proportion_train
    args = param.proportion_train_args()
    proportion_train.main(args)

    #proportion_test
    args = param.proportion_test_args()
    proportion_test.main(args)

    print("process finish")