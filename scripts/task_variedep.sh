#!/bin/bash

run_hsicbt -cfg config/varied-epoch.yaml -ep 1  -ei 1 -tt hsictrain
run_hsicbt -cfg config/varied-epoch.yaml -ep 5  -ei 2 -tt hsictrain
run_hsicbt -cfg config/varied-epoch.yaml -ep 10 -ei 3 -tt hsictrain
run_hsicbt -cfg config/varied-epoch.yaml -ep 1  -ei 1 -tt format
run_hsicbt -cfg config/varied-epoch.yaml -ep 1  -ei 2 -tt format
run_hsicbt -cfg config/varied-epoch.yaml -ep 1  -ei 3 -tt format
run_plot -t varied-epoch -dc mnist -e pdf
