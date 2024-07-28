# Restoration of Blurred and Hazy Images via Multi-Channel Frequency Domain Modulation

Zhehang Qiu ,Jie Zhou, Jianming Zhan, Huijuan Zhang


## Update
- 2024.07.26 	:Release the model and code

>Image restoration seeks to recover clear images from degraded ones, yet there is limited research focused on handling compound degradation. Currently, convolutional neural networks (CNNs) have achieved widely applied in image restoration. However, most methods focus on spatial information while overlooking frequency domain features. In this paper, we combine dual-domain information to explore the restoration of images with combined blur and haze degradation. First, we created the GoPro-haze dataset and analyzed the characteristics of degraded images from both spatial and frequency domains. Considering the increased demand for degradation information collection due to compound degradation, we introduced the Dual-Path Selective Frequency Module to achieve a larger receptive field. By integrating multichannel information in the spatial domain, we integrated the Multi-Channel Frequency-Domain Attention Module to facilitate modulation of different frequency information. Additionally, we employed transfer learning to enhance training effectiveness. Comprehensive experiments show that the proposed network BHNet performs well in both deblurring and dehazing tasks, significantly outperforming existing methods in compound blur and haze degradation scenarios.

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation
Please refer to respective directories.
## Results

| Task                    | Dataset      | RESULTS                                                           |
|-------------------------|--------------|-------------------------------------------------------------------|
| **Motion Deblurring**   | GOPRO        | [Baidu](https://pan.baidu.com/s/1ays0B3h2wMRVt_0ENuhFQQ?pwd=0lxr) |
|                         | HIDE         | [Baidu](https://pan.baidu.com/s/1tzApS11cpgldx5MaS26XYA?pwd=9jnz) |
| **Image Dehazing**      | SOTS-Indoor  | [Baidu](https://pan.baidu.com/s/1RpnWPzyGqdTBMbmYaXqfsg?pwd=8rta) |
|                         | SOTS-Outdoor | [Baidu](https://pan.baidu.com/s/1ZLpDmqcKbDNI_5DeLljw5Q?pwd=rmg1) |
| **Dehazing-Deblurring** | GoPro-haze   | [Baidu](https://pan.baidu.com/s/1QSlyAk_raOrgQ9pvuXQ9mQ?pwd=c0bh) |
## 
The proposed dataset GOPRO-haze can be download [Baidu](https://pan.baidu.com/s/1i95Nqctf_XMijfRaqKm8eA?pwd=oix0).
Please refer to respective directories.
~~~
## Acknowledgement
This project is mainly based on [MIMO-UNet](https://github.com/chosj95/MIMO-UNet) and [SFnet](https://github.com/c-yn/SFNet).

