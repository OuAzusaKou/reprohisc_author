#!/bin/bash

run_hsicbt -cfg config/general-hsicbt.yaml   -tt hsictrain -m resnet-conv -dc cifar10 -d 15 -mf hsic_weight_linear_cifar10.pt
run_hsicbt -cfg config/general-format.yaml   -tt format    -m resnet-conv -dc cifar10 -d 15 -mf hsic_weight_linear_cifar10.pt
run_hsicbt -cfg config/general-backprop.yaml -tt backprop  -m resnet-conv -dc cifar10 -d 15 -mf hsic_weight_linear_cifar10.pt
run_plot -t general -dc cifar10 -e pdf

