# FIBA: FEDERATED INVISIBLE BACKDOOR ATTACK

## Overview
This repository is an PyTorch Implementation for "FIBA: FEDERATED INVISIBLE BACKDOOR ATTACK (under review)".

**Authors**: Lu Zhang, Baolin Zheng \
**Codes**: https://github.com/LukeZane118/FIBA

Note: Other baselines are preparing...
## Environment

The code was developed and tested on the following python environment: 
```
python 3.8.8
torch 1.8.1
visdom 0.2.3
seaborn 0.11.1
torch_dct 0.1.5
opencv-python 4.6.0.66
scikit-learn 0.24.1
pyyaml 5.4.1
tensorboard 2.11.0
piq 0.7.1
python-hessian 1.1.2
```

## Usage
### Prepare the dataset:
#### Tiny-imagenet dataset:

- download the dataset [tiny-imagenet-200.zip](https://tiny-imagenet.herokuapp.com/) into dir `./utils` 
- reformat the dataset.
```
cd ./utils
./process_tiny_data.sh
```

#### Others:
CIFAR-10 will be automatically download.

### Reproduce experiments: 

- prepare the pretrained model:
Our pretrained clean models for attack follow the [DBA](https://openreview.net/forum?id=rkgyS0VFvr).

- we can use Visdom to monitor the training progress.
```
./visdom_run.bash
```

- run experiments:
```
./run.bash
```
Parameters can be changed in those yaml files to reproduce our experiments.


## Acknowledgement 
- [DBA](https://github.com/AI-secure/DBA)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
- [ISSBA](https://github.com/yuezunli/ISSBA)
- [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/ISSBA.py)
- [FTrojan](https://github.com/FTrojanAttack/FTrojan)
- [AdvDoor](https://github.com/ZQ-Struggle/AdvDoor)
- [trojanzoo](https://github.com/ain-soph/trojanzoo/blob/main/trojanvision/defenses/backdoor/input_filtering/strip.py)
- [TrojAi](https://github.com/lijiachun123/TrojAi)
