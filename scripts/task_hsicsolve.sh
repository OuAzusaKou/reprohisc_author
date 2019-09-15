#!/bin/bash

run_hsicbt -cfg config/hsicsolve.yaml -tt backprop  -dc mnist
run_hsicbt -cfg config/hsicsolve.yaml -tt hsictrain -dc mnist
run_plot -t hsic-solve -dc mnist -e pdf
run_hsicbt -cfg config/hsicsolve.yaml -tt backprop  -dc fmnist
run_hsicbt -cfg config/hsicsolve.yaml -tt hsictrain -dc fmnist
run_plot -t hsic-solve -dc fmnist -e pdf
run_hsicbt -cfg config/hsicsolve.yaml -tt backprop  -dc cifar10
run_hsicbt -cfg config/hsicsolve.yaml -tt hsictrain -dc cifar10
run_plot -t hsic-solve -dc cifar10 -e pdf

