#!/bin/bash


# the example running same experiment with different randomseed
for s in {1..10}
do
    echo ${s}, ${RANDOM}
    run_hsicbt -cfg config/needle.yaml -tt hsictrain -ep 5 -sd ${RANDOM}
    run_plot -t needle -dc mnist -e pdf -tt hsictrain -ft 'sigma:(20,20) random seed:'"$RANDOM"
done
