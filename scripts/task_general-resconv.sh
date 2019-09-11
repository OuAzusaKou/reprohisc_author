#!/bin/bash

run_hsicbt -cfg config/general-hsicbt.yaml   -tt hsictrain -m resconv -dc cifar10
run_hsicbt -cfg config/general-format.yaml   -tt format    -m resconv -dc cifar10
run_hsicbt -cfg config/general-backprop.yaml -tt backprop  -m resconv -dc cifar10
run_plot -t general -dc cifar10 -e pdf

