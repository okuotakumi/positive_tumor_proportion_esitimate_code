# Automatic positive tumor proportion diagnosis tests
![overview](./overview.png)
## Prerequisites
- PIL
- pytorch
- opencv
- openslide
- tqdm
- skimage

## Quick start

Running whole script
```
python main.py
```

## About code
|- base_detector : Detect all cells  
| 　|- datas : Sample data for train     
| 　|- model : CNN model code  
| 　|- preprocessing  
| 　| 　|- img_slice.py : code to separate core image into patch images    
|　 | 　|- patch2core.py : Code to obtain cell coordinates from the estimation results  
| 　|- utils : directory with dataloader code  
| 　|- basedetector_train.py : Train code for cell detection   
| 　L basedetector_pred.py : Code for estimation with learned weights  

|- cancer_or_noncancer_detection : Classify cells as tumor or non-tumor  
| 　|- datas : Sample data for train  
| 　|- model : CNN model code      
| 　|- preprocessing  
| 　| 　|- point2patch : Code to adapt from core image coordinates to patch image coordinates  
| 　|- utils : directory with dataloader code  
| 　|- c_or_n_train.py : Train codes for tumor and non-tumor classification  
| 　L c_or_n_pred.py : Code for estimation with learned weights  

|- estimation_proportion : Estimating the Positive Tumor Proportion  
| 　|- datas : Sample data for train  
| 　|- model : CNN model code + self-made loss function   
| 　|- preprocessing  
| 　| 　|- img_resize.py : Code to resize core image to input image size  
|　 | 　|- make_mask.py : Code to create a mask from core image and tumor cell coordinates  
| 　|- utils : directory with dataloader code  
| 　|- proportion_train.py : Code to train positive tumor proportion  
| 　L proportion_test.py : Code to output and evaluate the positive tumor proportion  

|- sample_ica : Remove color for cell detection.  
| 　L sample_ica.py : Code for removing color from an image  

|- env_file_yml : Environment used  

|- param.py : Parameters for batch  

|- main.py : Batch positive tumor proportion estimation code  
