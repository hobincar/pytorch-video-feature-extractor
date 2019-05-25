# pytorch-video-feature-extractor

Extract video features from pre-trained CNNs.


## Environment

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Nvidia Geforce GTX Titan Xp 12GB
* Python 3.6.8


## Supported CNNs

* AlexNet
* GoogleNet
* VGGNet
* DenseNet
* MobileNet
* ResNet
* ShuffleNet
* SqueezeNet
* InceptionNet (v3 & v4)

## How to use

### Step 1. Install python packages specified in `requirements.txt`

```
$ python -m venv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```

### Step 2. Prepare videos and put them in to a directory.

For example, for my case

```
|-- datasets
    |-- MSVD
        |-- C9LHk0AJI7U_196_205.avi
        |-- D1tTBncIsm8_248_254.avi
        ...
```

### Step 3. Extract CNN features from videos.

For example, to extract VGG19 features from 'MSVD' dataset,

```
(.env) $ python main.py \
    --video_dpath datasets/MSVD \
    --model vgg19_bn \
    --batch_size 25 \
    --stride 5 \
    --out features/MSVD_VGG19.hdf5
```

, or you can refer to the script files in the `scripts` directory.


## References

* AlexNet: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
* GoogleNet: https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py
* VGGNet: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
* DenseNet: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
* MobileNet: https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
* ResNet: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
* ShuffleNet: https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
* SqueezeNet: https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py
* InceptionNet-v3: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
* InceptionNet-v4: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py


## Acknowlegement

I got the codes and pre-trained models from
[pytorch/vision](https://github.com/pytorch/vision), and [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch).
Many thanks!
