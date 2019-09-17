# Task Script

Each config file represents the single task, which might have several trainings according to the paper figures. You could simply run `batch.sh` to reproduce all the figures from our paper, or either go to each task script described in batch.sh for further research.


# General Procedure

#### pre-action
- go to the project root directory
- setting the environment `source env.sh`
- run command `task_`

#### runtime
- load config `[config_path]`
- training, and save the logs under `./assets/logs` (optional)


# Tasks

#### varied-activation (fig2a-c)
- note
- commands
```sh
run_hsicbt -cfg config/varied-activation.yaml 
```

#### varied-depth (fig2d-f)
- note
- commands
```sh
run_hsicbt -cfg config/varied-depth.yaml 
```

#### needle (fig3)
- note
- commands
```sh
run_hsicbt -cfg config/needle.yaml 
```

#### hsicsolve (fig4, fig5)
- note
- commands
```sh
run_hsicbt -cfg config/hsicsolve.yaml 
```

#### varied-epoch (fig6a-b)
- note
- commands
```sh
run_hsicbt -cfg config/varied-epoch.yaml 
```

#### varied-dim (fig7a)
- note
- commands
```sh
run_hsicbt -cfg config/varied-dim.yaml
```

#### sigma-combined (fig7b)
- note
- commands
```sh
run_hsicbt -cfg config/sigma-combined.yaml
```
