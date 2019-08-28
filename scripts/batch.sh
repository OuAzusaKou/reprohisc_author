#!/bin/bash

# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#	General experiment: 
# 		HSIC-Bottleneck training and the comparison of format/standard training
#		will produce train/valid performance comparison
# # # # # # # # # # # # # # # # # # # # # # # # # # # 

# ResNet with linear model
run_hsicbt -cfg config/hsictrain-reslinear.yaml
run_hsicbt -cfg config/general-reslinear.yaml

# Linear model
run_hsicbt -cfg config/hsictrain-linear.yaml
run_hsicbt -cfg config/general-linear.yaml

# ResNet with Conv model
run_hsicbt -cfg config/hsictrain-resconv.yaml
run_hsicbt -cfg config/general-resconv.yaml

# Conv model
run_hsicbt -cfg config/hsictrain-conv.yaml
run_hsicbt -cfg config/general-conv.yaml


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#	HSIC-Bottleneck solve:
#		pure HSIC-Bottleneck solve the classification, compared with standard training
# 		will produce one-hot activation footage and training acc comparison
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
run_hsicbt -cfg config/hsicsolve.yaml


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#	Niddle:
#		a niddle model that making single scalar output, this will show the comparison
#		of class activation distribution separation between HISC-Bottleneck/standard
# # # # # # # # # # # # # # # # # # # # # # # # # # # 
run_hsicbt -cfg config/niddle.yaml