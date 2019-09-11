#!/bin/bash

run_hsicbt -cfg config/hsicsolve.yaml -tt backprop
run_hsicbt -cfg config/hsicsolve.yaml -tt hsictrain
run_plot -t hsic-solve -dc mnist -e pdf
