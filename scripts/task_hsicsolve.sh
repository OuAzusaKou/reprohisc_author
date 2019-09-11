#!/bin/bash

run_hsicbt -cfg config/hsicsolve-bp.yaml
run_hsicbt -cfg config/hsicsolve-ht.yaml
run_plot -t hsic-solve -dc mnist -e pdf
