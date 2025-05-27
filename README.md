# ltx_lora_training_i2v_t2v
Lora traing script for [Lightricks/LTX-video](https://github.com/Lightricks/LTX-Video) modified from [a-r-r-o-w/finetrainers](https://github.com/a-r-r-o-w/finetrainers) to support image to video training and STG guidancec for inference;
tested on two RTX 4090 24GB

I will update this README add more details onece i finished meta's [VideoJAM](https://hila-chefer.github.io/videojam-paper.github.io/) training method

## Bullet time effects lora results
| Image           | Video      |
| ----------- | ----------- |
| ![action111-](https://github.com/user-attachments/assets/e694bbbe-305d-43d9-a355-d9d9d1bef3bd) | <video src="https://github.com/user-attachments/assets/02998797-8172-4246-84cb-1ac3d31a60ab"></video>|
| ![action140-1](https://github.com/user-attachments/assets/3ff27d6f-7015-4ede-93ad-571c1b78061b) | <video src="https://github.com/user-attachments/assets/2f3288a3-5d7b-481e-9af5-8c24ff3af47a"></video>|
| ![action52-](https://github.com/user-attachments/assets/faf80cb0-9419-4055-bc92-bc61acc15485) | <video src="https://github.com/user-attachments/assets/8138f365-2fa8-49e9-8491-8f0858d42fd9"></video> |


## Precompute 
```
python precompute.py
```
you can check if your precomputed pt is correct in this file: **"test_rebuild_from_precomputed.py"** please edit this file 
```
python test_rebuild_from_precomputed.py
```


## Prepare the config file
edit **configs/ltx.yaml** 

## Start Training with two GPUs
```
accelerate launch --config_file ./configs/uncompiled_2.yaml ltx_train.py 
```

## Run inference with trained Lora
edit **batch_gen_i2v_STG.py** set lora path and output dirs
```
python batch_gen_i2v_STG.py
```



















