#!/bin/bash

# # # # # # # # # # # # # # # 
# Batch-running: 
#	please see config/README.md for more information
# # # # # # # # # # # # # # # 

# genernal
task_general.sh

# fig2a-c
task_varied-act.sh

# fig2d-f
task_varied-depth.sh

# fig3
task_needle.sh

# fig4-5
task_hsicsolve.sh

# fig6-b
task_varied-ep.sh

# fig7a
task_varied-dim.sh

# fig7b
task_combsig.sh

# fig8 `command [dataset] [hsictrain_epoch] [is_hsictrain]`
task_general-resconv.sh cifar10 50 1





