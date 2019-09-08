#!/bin/bash

# # # # # # # # # # # # # # # 
# Batch-running: 
#	please see config/README.md for more information
# # # # # # # # # # # # # # # 

run_hsicbt -cfg config/hsictrain-reslinear.yaml
run_hsicbt -cfg config/general-reslinear.yaml

# fig2a-c
run_hsicbt -cfg config/varied-activation.yaml 

# fig2d-f
run_hsicbt -cfg config/varied-depth.yaml 

# fig3
run_hsicbt -cfg config/needle.yaml 

# fig4-5
run_hsicbt -cfg config/hsicsolve.yaml 

# fig6-b
run_hsicbt -cfg config/varied-epoch.yaml 

# fig7a
run_hsicbt -cfg config/varied-dim.yaml

# fig7b
run_hsicbt -cfg config/sigma-combined.yaml