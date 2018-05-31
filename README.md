## Reid project framework
this is a general person reid project framework for undergraduate members in iSee lab.
many useful tools are included.
this repo implements a baseline id-discriminative model with a backbone ResNet-50,
which is effectively a standard image classification network,
where the classes are person identities.

#### environment
- python 3.6, pytorch 0.4.0, cuda 8, matlab, [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)
- assuming you have a gpu (code is not device adaptive), running on a linux server

#### data
- run ./data/make_imdb_Duke.m and ./data/make_imdb_Market.m in MATLAB to organize the datasets.
don't forget to replace the *dir_path* with yours, which should contain the original dataset.
- [optionally] download the pretrained model parameter from https://download.pytorch.org/models/resnet50-19c8e357.pth
and put it into ./data/
- if still any problem, please contact me for the wrapped data.
- tip: in practice you don't wanna debug with a full dataset due to loading time consuming.
run data/make_imdb_Market_debug.m for a mini version to speed up testing and debugging.

#### running
- if you run on gpu 0, type
```bash
cd src
python src/main.py --gpu 0 --save_path runs
```
the second argument is the save_path where you wanna save logs and checkpoints.
note that you should put in save_path a setting file args.yaml,
which can be modified from runs/args.yaml
in practice, you may wanna distribute different setting files and organize some directories in ./runs,
for organizing your experiments.

please contact me at xKoven@gmail.com if any problem.