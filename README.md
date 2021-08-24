# plantcv-labeling-tools

### A human-in-the-loop labeling tool

The labeling is based on maskRCNN instance segmentation result

There are two main steps:
1. Remove redundant and/or add missed
2. Label time series

Steps:
1. Download instance segmentation results (from server) and save to local computer

e.g.:
  - From: /shares/mgehan_share/hsheng/projects/URoLE/instance_seg/10.9.1.241_wtCol_lowCo2_BR5/hard_crop/seg/index0
  - To: /Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant0/segmentation

2. Download original images (cropped, from server) and save to local computer

e.g.:
  - From: /shares/mgehan_share/hsheng/projects/URoLE/crop/crop_results/10.9.1.241_wtCol_lowCo2_BR5/plant0/hard_crop_450_450/selected
  - To: /Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant0/images

3. Remove redundant segments and/or add missed segments
- Set `dir_img`and `dir_sg` to be the directories of images and segmentation
- (Option) Set desired directories to save label images (index images), curated segmentation results, curated label images
  by setting `dir_im_lb`, `dir_seg_cure`, `dir_lb_im_cure`. 
- Run "manual_rem_add.py":
  
  In a terminal, type
  `python manual_rem_add.py`
- Note: if zoom-in is needed, zoom in first using the matplotlib zoom-in tool (the magnifying glass), then click the 
  "Start" button or hit on "a" to start drawing.
   
4. Run "manual_labeling.py"
- Set `dir_img`, `dir_cure_lb_im`and `dir_gt`
- Run "manual_rem_add.py":
  In a terminal, type
  `python manual_labeling.py`
- Note 1: If zoom-in is needed, zoom in first using the matplotlib zoom-in tool (the magnifying glass), then click the 
  "Start" button or hit on "a" to start drawing.
- Note 2: If find some issue before click on the "Next" button (or hit on "n"), "start over" feature is available by 
  clicking on the "Start Over" button.
  
5. Move generated ground-truth files back to the server

e.g.: 
  - From: /Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/ground_truth/plant0
  - To: /shares/mgehan_share/hsheng/projects/URoLE/ground_truth/10.9.1.241_wtCol_lowCo2_BR5/plant0

Also move index images, curated segmentation, curated index images to the server

e.g.: 
- index images
  - From: /Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant0/seg_labels
  - To: /shares/mgehan_share/hsheng/projects/URoLE/instance_seg/10.9.1.241_wtCol_lowCo2_BR5/hard_crop/seg_labels/plant0
  
- curated segmentation
  - From: /Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant0/curated_segmentation
  - To: /shares/mgehan_share/hsheng/projects/URoLE/instance_seg/10.9.1.241_wtCol_lowCo2_BR5/hard_crop/curated_seg/plant0

- curated index images
  - From: /Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant0/curated_seg_labels
  - To: /shares/mgehan_share/hsheng/projects/URoLE/instance_seg/10.9.1.241_wtCol_lowCo2_BR5/hard_crop/curated_seg_labels/plant0
  
6. Finally, we need to generate the time-series tracking ground-truth that can be used in the `evaluation` functions of 
   the `plantcv`'s `time_series_linking` subpackage.
   
   Find the notebook and follow the instructions here:
/shares/mgehan_share/hsheng/projects/URoLE/code/time_series_ground_truth/generate_gt_protocol.ipynb