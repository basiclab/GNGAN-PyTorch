# Gradient Normalization for Generative Adversarial Networks

Yi-Lun Wu, Hong-Han Shuai, Zhi-Rui Tam, Hong-Yu Chiu

Paper: [https://arxiv.org/abs/2109.02235](https://arxiv.org/abs/2109.02235)

This is the official implementation of Gradient Normalized GAN (GN-GAN).

## Requirements
- Python 3.8.9
- Python packages
    ```sh
    # update `pip` for installing tensorboard.
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10

    Pytorch build-in CIFAR-10 will be downloaded automatically.

- STL-10

    Pytorch build-in STL-10 will be downloaded automatically.

- CelebA-HQ 128/256

    We obtain celeba-hq from [this repository](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download) and preprocess it into `lmdb` file.
    - 256x256
        ```
        python dataset.py path/to/celebahq/256 ./data/celebahq/256
        ```
    - 128x128

        We split data into train test splits by filenames, the test set contains images from `27001.jpg` to `30000.jpg`.
        ```
        python dataset.py path/to/celebahq/128/train ./data/celebahq/128
        ```
    The folder structure:
    ```
    ./data/celebahq
    ├── 128
    │   ├── data.mdb
    │   └── lock.mdb
    └── 256
        ├── data.mdb
        └── lock.mdb
    ```

- LSUN Church Outdoor 256x256 (training set)

    The folder structure:
    ```
    ./data/lsun/church/
    ├── data.mdb
    └── lock.mdb
    ```

## Preprocessing Datasets for FID
Pre-calculated statistics for FID can be downloaded [here](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing):
- cifar10.train.npz - Training set of CIFAR10
- cifar10.test.npz - Testing set of CIFAR10
- stl10.unlabeled.48.npz - Unlabeled set of STL10 in resolution 48x48
- celebahq.3k.128.npz - Last 3k images of CelebA-HQ 128x128
- celebahq.all.256.npz - Full dataset of CelebA-HQ 256x256
- church.train.256.npz - Training set of LSUN Church Outdoor

Folder structure:
```
./stats
├── celebahq.3k.128.npz
├── celebahq.all.256.npz
├── church.train.256.npz
├── cifar10.test.npz
├── cifar10.train.npz
└── stl10.unlabeled.48.npz
```

**NOTE**

All the reported values (Inception Score and FID) in our paper are calculated by official implementation instead of our implementation. 


## Training
- Configuration files
    - We use `absl-py` to parse, save and reload the command line arguments.
    - All the configuration files can be found in `./config`. 
    - The compatible configuration list is shown in the following table:

        |Script           |Configurations|Multi-GPU|
        |-----------------|--------------|:-------:|
        |`train.py`       |`GN-GAN_CIFAR10_CNN.txt`<br>`GN-GAN_CIFAR10_RES.txt`<br>`GN-GAN_CIFAR10_BIGGAN.txt`<br>`GN-GAN_STL10_CNN.txt`<br>`GN-GAN_STL10_RES.txt`<br>`GN-GAN-CR_CIFAR10_CNN.txt`<br>`GN-GAN-CR_CIFAR10_RES.txt`<br>`GN-GAN-CR_CIFAR10_BIGGAN.txt`<br>`GN-GAN-CR_STL10_CNN.txt`<br>`GN-GAN-CR_STL10_RES.txt`||
        |`train_ddp.py`|`GN-GAN_CELEBAHQ128_RES.txt`<br>`GN-GAN_CELEBAHQ256_RES.txt`<br>`GN-GAN_CHURCH256_RES.txt`|:heavy_check_mark:|

- Run the training script with the compatible configuration, e.g.,
    - `train.py` supports training gan on `CIFAR10` and `STL10`, e.g.,
        ```sh
        python train.py \
            --flagfile ./config/GN-GAN_CIFAR10_RES.txt
        ```
    - `train_ddp.py` is optimized for multi-gpu training, e.g.,
        ```
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ddp.py \
            --flagfile ./config/GN-GAN_CELEBAHQ256_RES.txt
        ```

- Generate images from checkpoints, e.g.,

    `--eval`: evaluate best checkpoint.

    `--save PATH`: save the generated images to `PATH`
    ```
    python train.py \
        --flagfile ./logs/GN-GAN_CIFAR10_RES/flagfile.txt \
        --eval \
        --save path/to/generated/images
    ```

## How to integrate Gradient Normalization into your work?
The function `normalize_gradient` is implemented based on `torch.autograd` module, which can easily normalize your forward propagation of discriminator by updating a single line.
```python
from torch.nn import BCEWithLogitsLoss
from models.gradnorm import normalize_gradient

net_D = ...     # discriminator
net_G = ...     # generator
loss_fn = BCEWithLogitsLoss()

# Update discriminator
x_real = ...                                    # real data
x_fake = net_G(torch.randn(64, 3, 32, 32))      # fake data
pred_real = normalize_gradient(net_D, x_real)   # net_D(x_real)
pred_fake = normalize_gradient(net_D, x_fake)   # net_D(x_fake)
loss_real = loss_fn(pred_real, torch.ones_like(pred_real))
loss_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))
(loss_real + loss_fake).backward()              # backward propagation
...

# Update generator
x_fake = net_G(torch.randn(64, 3, 32, 32))      # fake data
pred_fake = normalize_gradient(net_D, x_fake)   # net_D(x_fake)
loss_fake = loss_fn(pred_fake, torch.ones_like(pred_fake))
loss.backward()                                 # backward propagation
...

```

## Citation
If you find our work is relevant to your research, please cite:
```
@InProceedings{GNGAN_2021_ICCV,
    author = {Yi-Lun Wu, Hong-Han Shuai, Zhi Rui Tam, Hong-Yu Chiu},
    title = {Gradient Normalization for Generative Adversarial Networks},
    booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    month = {Oct},
    year = {2021}
}
```