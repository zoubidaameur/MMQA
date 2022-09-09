### MMQA: Multi-Task Learning for Multi-distance Image Quality Assessment
Zoubida Ameur, Sid Ahmed Fezza and Wassim Hamidouche


##Abstract
With the increasing use of image-based applications, users are constantly demanding higher quality images. Multimedia devices and services must therefore be able to meet their requirements. Accordingly, they must be able to evaluate the quality of images efficiently and reliably so they can then improve it. Images are viewed from various devices, resulting in their perceived quality being highly dependent on the device as well as the viewing distance. In this paper, we present a novel image quality metric (IQM) that assesses objectively the perceived quality of an image considering the viewing distance. Our proposed metric is a deep multi-task learning model composed of a pretrained convolutional neural network followed by N parallel networks of fully connected layers. It takes as input a single image and outputs N different quality scores corresponding to N
different viewing distances. We evaluate the proposed approach on colourlab image database image quality (CIDIQ), multi-distance laboratory for image and video engineering (M-LIVE) and viewing distance-changed image database (VDID) databases. Our model shows superior performance to state-of-the-art single and multiple viewing distance metrics.

##Network architecture
![](model.pdf)


##Structure of directory
- `data.py`: data loader
- `model.py`: models builder
- `train.py`: training loops.
- `main.py`: main script to start training
- `test.py`: script for the evaluation of a trained model on a test dataset


## Training : 
To train the model, run the following command :
```bash
nvidia-smi 
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset CIDIQ --batch_size 4 --epochs 30
```
## Testing:
To test a trained model on test-set, run the following:
```bash
nvidia-smi  #to see free nodes 
CUDA_VISIBLE_DEVICES=2 python3 test.py --dataset CIDIQ

##Citation

