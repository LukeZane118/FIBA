#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main.py --params configs/Multi-Shot-Attack/CIFAR10_FIBA_Params.yaml --postfix [FIBA][Multi-Attack][Mask_0.95]