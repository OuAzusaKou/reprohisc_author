# Configuration

Each config file represents the single task, which might have several trainings to have the comparisons. After the training, you might switch off the flags like "do_training", "do_training_hsic", "do_training_format" according to tasks. This gives the opportunity to modify other functionalities replied on the output logs without re-train again.

# General Procedure

#### pre-action
- go to the project root directory
- setting the environment `source env.sh`
- run command `run_hsicbt -cfg [config_path]`

#### runtime
- load config `[config_path]`
- training, and save the logs under `./assets/logs` (optional)
- load the logs under `./assets/log`
- plot and save under `./assets/exp`


# Tasks

#### varied-activation (fig2a-c)
- note
- commands
```sh
run_hsicbt -cfg config/varied-activation.yaml 
```
- outputs
```sh
./assets/exp/fig2a-varied-activation-hsic_xz-mnist.pdf
./assets/exp/fig2b-varied-activation-hsic_yz-mnist.pdf
./assets/exp/fig2c-varied-activation-acc-mnist.pdf
```

#### varied-depth (fig2d-f)
- note
- commands
```sh
run_hsicbt -cfg config/varied-depth.yaml 
```
- outputs
```sh
./assets/exp/fig2d-varied-depth-hsic_xz-mnist.pdf
./assets/exp/fig2e-varied-depth-hsic_yz-mnist.pdf
./assets/exp/fig2f-varied-depth-acc-mnist.pdf
```

#### needle (fig3)
- note
- commands
```sh
run_hsicbt -cfg config/needle.yaml 
```
- outputs
```sh
./assets/exp/fig3b-needle-1d-dist-hsic.pdf
./assets/exp/fig3a-needle-1d-dist-standard.pdf
```

#### hsicsolve (fig4, fig5)
- note
- commands
```sh
run_hsicbt -cfg config/hsicsolve.yaml 
```
- outputs
```sh
./assets/exp/fig4-hsic-solve-actdist-mnist.pdf
./assets/exp/fig5-hsic-solve-mnist-linear-train-acc.pdf
```

#### varied-epoch (fig6a-b)
- note
- commands
```sh
run_hsicbt -cfg config/varied-epoch.yaml 
```
- outputs
```sh
./assets/exp/fig6a-varied-epoch-acc-mnist.pdf
./assets/exp/fig6b-varied-epoch-loss-mnist.pdf
```

#### varied-dim (fig7a)
- note
- commands
```sh
run_hsicbt -cfg config/varied-dim.yaml
```
- outputs
```sh
./assets/exp/fig7a-varied-dim-acc-mnist.pdf
```

#### sigma-combined (fig7b)
- note
- commands
```sh
run_hsicbt -cfg config/sigma-combined.yaml
```
- outputs
```sh
./assets/exp/fig7b-sigma-combined-mnist-linear-sigmacomb-train-acc.pdf
```
