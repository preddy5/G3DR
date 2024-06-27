
# G3DR: Generative 3D Reconstruction in ImageNet (CVPR 2024)
<a href="https://preddy5.github.io/g3dr_website/" target="_blank">[Project page]</a>
<a href="https://arxiv.org/abs/2403.00939" target="_blank">[Paper]</a>

![ImageNet samples 256x256](images/0_ImageNet00.gif)

Code for reproducing f_trigen network training and inference.

## Prepare config file
Config file for training on imagenet data is available in ./config/
Please update 'dataset_folder', 'renderer_path', 'save_dir' fields before training.

dataset_folder - path to the imagenet dataset along with depth, you can generate the dataset yourself by downloading imagenet and using an off-the-shelf monocular depth estimation network or download it from here: https://drive.google.com/drive/folders/1yAMr1Us9gD6F5P0lCd5qiouyZ9gT5P_n

renderer_path - path to eg3d library, clone/download https://github.com/NVlabs/eg3d/ and provide the path to the repo here. 

save_dir - path to dir to save the training logs.

## Requirements
Install Clip repo from https://github.com/openai/CLIP

All the other required libraries should be available to download via pip.

## Training
after configuring accelerate use the below command for training 
```
accelerate launch train.py --config config/imagenet_train.yaml
```  

for training with a single GPU without accelerate use 
```
python train.py --config config/imagenet_train.yaml
```
In case of errors replace "model.module" with "model"

## Pre-trained model
Pretrained f_trigen on imagenet can be found here: https://drive.google.com/file/d/1Bg5k3IYquph-cZbWJVW0A4kyyd-t7n-d/view 

Pretrained super-resolution model weights can be found here: https://drive.google.com/file/d/1Wsa0bbw_oP80O5DdCyE5qIQGunXxwgt7/view

## Visualization
```
CUDA_VISIBLE_DEVICES=0 python visualize.py --load_model ./checkpoint_generic.pt --config ./config/test.yaml --folder ./images/1/
```
The expected outputs are present in the output folder.



## Bibtex
Please consider citing our paper.
```
@inproceedings{reddy2024g3dr,
  title={G3DR: Generative 3D Reconstruction in ImageNet},
  author={Reddy, Pradyumna and Elezi, Ismail and Deng, Jiankang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9655--9665},
  year={2024}
}
```
