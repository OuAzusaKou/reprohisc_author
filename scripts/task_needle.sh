#!/bin/bash

#run_hsicbt -cfg config/needle.yaml -tt hsictrain
#run_hsicbt -cfg config/needle.yaml -tt backprop
#run_plot -t needle -dc mnist -e pdf -tt hsictrain
#run_plot -t needle -dc mnist -e pdf -tt backprop


run_hsicbt -cfg config/needle.yaml -tt hsictrain -sd 1234 -ep 1
run_plot -t needle -dc mnist -e pdf -tt hsictrain
mv ./assets/exp/fig3b-needle-1d-dist-hsictrain.pdf ./assets/exp/fig3b-needle-1d-dist-hsictrain-1234-1.pdf

run_hsicbt -cfg config/needle.yaml -tt hsictrain -sd 1234 -ep 1
run_plot -t needle -dc mnist -e pdf -tt hsictrain
mv ./assets/exp/fig3b-needle-1d-dist-hsictrain.pdf ./assets/exp/fig3b-needle-1d-dist-hsictrain-1234-2.pdf


#run_hsicbt -cfg config/needle.yaml -tt hsictrain -sd 1235
#run_plot -t needle -dc mnist -e pdf -tt hsictrain
#mv ./assets/exp/fig3b-needle-1d-dist-hsictrain.pdf ./assets/exp/fig3b-needle-1d-dist-hsictrain-1235.pdf
#
#run_hsicbt -cfg config/needle.yaml -tt hsictrain -sd 1236
#run_plot -t needle -dc mnist -e pdf -tt hsictrain
#mv ./assets/exp/fig3b-needle-1d-dist-hsictrain.pdf ./assets/exp/fig3b-needle-1d-dist-hsictrain-1236.pdf
#
#run_hsicbt -cfg config/needle.yaml -tt hsictrain -sd 1237
#run_plot -t needle -dc mnist -e pdf -tt hsictrain
#mv ./assets/exp/fig3b-needle-1d-dist-hsictrain.pdf ./assets/exp/fig3b-needle-1d-dist-hsictrain-1237.pdf
#
#run_hsicbt -cfg config/needle.yaml -tt hsictrain -sd 1238
#run_plot -t needle -dc mnist -e pdf -tt hsictrain
#mv ./assets/exp/fig3b-needle-1d-dist-hsictrain.pdf ./assets/exp/fig3b-needle-1d-dist-hsictrain-1238.pdf
#
