#!/bin/bash

run_hsicbt -cfg config/general-hsicbt.yaml   -tt hsictrain -m resnet-conv -dc cifar10 -d 15 -mf hsic_weight_linear_cifar10.pt --epochs 10 -ld 50 -lr 0.0005 -lhw 256 -vb
run_hsicbt -cfg config/general-format.yaml   -tt format    -m resnet-conv -dc cifar10 -d 15 -mf hsic_weight_linear_cifar10.pt -ep 1 -lr 0.01
run_hsicbt -cfg config/general-backprop.yaml -tt backprop  -m resnet-conv -dc cifar10 -d 15 -ep 1 -lr 0.01
run_plot -t general -dc cifar10 -e pdf

