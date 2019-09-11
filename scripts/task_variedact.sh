#!/bin/bash

run_hsicbt -cfg config/varied-activation.yaml -ei 1 -at relu    -tt hsictrain
run_hsicbt -cfg config/varied-activation.yaml -ei 2 -at tanh    -tt hsictrain
run_hsicbt -cfg config/varied-activation.yaml -ei 3 -at elu     -tt hsictrain
run_hsicbt -cfg config/varied-activation.yaml -ei 4 -at sigmoid -tt hsictrain
run_hsicbt -cfg config/varied-activation.yaml -ei 1 -at relu    -tt format
run_hsicbt -cfg config/varied-activation.yaml -ei 2 -at tanh    -tt format
run_hsicbt -cfg config/varied-activation.yaml -ei 3 -at elu     -tt format
run_hsicbt -cfg config/varied-activation.yaml -ei 3 -at sigmoid -tt format
run_plot -t varied-activation -dc mnist -e pdf
