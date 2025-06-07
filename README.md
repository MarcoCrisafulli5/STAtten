# STAtten: Spiking Transformer with Spatial-Temporal Attention [[Paper]](https://arxiv.org/pdf/2409.19764)
### ***Conference on Computer Vision and Pattern Recognition (CVPR), 2025***

![method.png](images/method.png)

[//]: # ([![License]&#40;https://img.shields.io/badge/License-MIT-yellow.svg&#41;]&#40;LICENSE&#41;)



[//]: # (## Table of Contents)

[//]: # (- [Overview]&#40;#overview&#41;)

[//]: # (- [Paper]&#40;#paper&#41;)

[//]: # (- [Installation]&#40;#installation&#41;)

[//]: # (- [Usage]&#40;#usage&#41;)

[//]: # (- [Training Details]&#40;#training-details&#41;)

[//]: # (- [Dependencies]&#40;#dependencies&#41;)

[//]: # (- [Contributing]&#40;#contributing&#41;)

[//]: # (- [Citation]&#40;#citation&#41;)

## Abstract
Spike-based Transformer presents a compelling and energy-efficient alternative to traditional Artificial Neural Network (ANN)-based Transformers, achieving impressive results through sparse binary computations. However, existing spike-based transformers predominantly focus on spatial attention while neglecting crucial temporal dependencies inherent in spike-based processing, leading to suboptimal feature representation and limited performance. To address this limitation, we propose Spiking Transformer with **S**patial-**T**emporal **Atten**tion (**STAtten**), a simple and straightforward architecture that efficiently integrates both spatial and temporal information in the self-attention mechanism. STAtten introduces a block-wise computation strategy that processes information in spatial-temporal chunks, enabling comprehensive feature capture while maintaining the same computational complexity as previous spatial-only approaches. Our method can be seamlessly integrated into existing spike-based transformers without architectural overhaul. Extensive experiments demonstrate that STAtten significantly improves the performance of existing spike-based transformers across both static and neuromorphic datasets, including CIFAR10/100, ImageNet, CIFAR10-DVS, and N-Caltech101. 

## Dependency
- timm == 0.6.12
- pytorch == 1.13.1
- cupy
- spikingjelly == 0.0.0.0.12
- tensorboard

## Data 
- CIFAR10-DVS: Download through Spikingjelly framework
- N-Caltech101: Download through Spikingjelly framework
- ImageNet: https://www.image-net.org/index.php

   ```bash
   conf
   ├── ...
   data
   ├── CIFAR10/100
   │   └── ...
   ├── CIFAR10-DVS
   │   ├── frames_number_10_split_by_number
   │   └── frames_number_16_split_by_number
   ├── N-Caltech101
   │   ├── frames_number_10_split_by_number
   │   └── frames_number_16_split_by_number
   ├── ImageNet
   │   └── ...


## Usage
### Train
Replace the "-c" for setting the configurations
- **CIFAR10/100**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py -c conf/cifar10/2_512_200E_t4.yml --model sdt --spike-mode lif --attention_mode STAtten
- **CIFAR10-DVS**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py -c conf/cifar10-dvs/2_256_200E_t16_TET.yml --model sdt --spike-mode lif --attention_mode STAtten
- **N-Caltech101**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py -c conf/ncaltech101/2_256_200E_t16.yml --model sdt --spike-mode lif --attention_mode STAtten
- **ImageNet**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 train.py -c conf/imagenet/8_768_200E_t4.yml --model sdt --spike-mode lif --attention_mode STAtten

### Test
Replace the "--resume" with your trained checkpoints
- **ImageNet**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29500 test.py -c conf/imagenet/8_768_200E_t4.yml --model sdt --spike-mode lif --attention_mode STAtten --resume {checkpoints} --no-resume-opt

## Ackonwledgements
Our code is based on below repositories. Thank you for your valuable works!
- Spike-driven Transformer: https://github.com/BICLab/Spike-Driven-Transformer
- Spikingjelly: https://github.com/fangwei123456/spikingjelly

## Citation <span style="color:blue">📚</span>
If you find our "Spiking Transformer with Spatial-Temporal Attention" paper useful or relevant to your research, please kindly cite our paper:
  ```bash
  @inproceedings{lee2025spiking,
  title={Spiking transformer with spatial-temporal attention},
  author={Lee, Donghyun and Li, Yuhang and Kim, Youngeun and Xiao, Shiting and Panda, Priyadarshini},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13948--13958},
  year={2025}
}
