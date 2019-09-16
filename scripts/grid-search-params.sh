#!/bin/bash

# please refer to bin/run_hsicbt to see the tuneable parameters,
# for this tutorial, sigma and batch_size are tested
p_sigma=(10 20 30)
p_batchsize=(128 256)

# unfortunately, the nested loopping is the current solution. will improve it later
for p_s in ${p_sigma[@]}
do
    for p_bs in ${p_batchsize[@]}
    do
        run_hsicbt -cfg config/needle.yaml -tt hsictrain -ep 5 -bs ${p_bs} -s ${p_s}
        run_plot -t needle -dc mnist -e pdf -tt hsictrain -ft 'sigma:('"${p_s}"','"${p_s}"'); batch_size:'"${p_bs}"
    done
done

