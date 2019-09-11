#!/bin/bash

run_hsicbt -cfg config/needle.yaml -tt hsictrain
run_hsicbt -cfg config/needle.yaml -tt backprop
run_plot -t needle -dc mnist -e pdf -tt hsictrain
run_plot -t needle -dc mnist -e pdf -tt backprop
