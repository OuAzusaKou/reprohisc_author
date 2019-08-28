# HSIC-Bottleneck


## Environment
- pytorch-1.1.0
- numpy-1.16.4
- scipy-1.3.0
- tqdm-4.33.0

## Usage
Depending on the task, the command will run all experiments and save the figure into assets folder

#### at the beginning
- go to project root
- environment:
```sh
source env.sh
```
- Then do the following tasks, where each task produces a couple of experimental images under assets folder. For the comparison, please also see the images produced in advanced in the assets/samples folder.

#### General
- The general comparison between our work and standard training
- will create assets/standard-hsic-comparison.jpg
```sh
run_hsicbt -cfg config/hsictrain.yaml # making HSIC-Bottleneck first, which will be loaded in general
run_hsicbt -cfg config/general.yaml
```
- experiment sample (general comparison between format-train and standard-train)
<img src="./assets/samples/standard-hsic-comparison.jpg"  width="256" height="256">
<img src="./assets/samples/standard-hsic-comparison-epoch-test-acc.jpg"  width="256" height="256">
<img src="./assets/samples/standard-hsic-comparison-epoch-train-acc.jpg"  width="256" height="256">

#### HSICSolve
- Pure HSIC solving the classification problem
- model 784-256-256-256-256-256-10
- will create assets/hsic-solve-actdist.jpg, for one-hot activation visualization
- will create assets/hsic-solve-comparison.jpg, for comparing the backprop
```sh
run_hsicbt -cfg config/hsicsolve.yaml
```
- experiment sample (left:hsic/standard comparison, right: one-hot hsic activation)
<img src="./assets/samples/hsic-solve-comparison.jpg"  width="256" height="256">
<img src="./assets/samples/hsic-solve-actdist.jpg"  width="256" height="256">

#### Niddle
- 1d output network to plot the activation distribution, in order to visualize how HSIC-bottleneck separate the class signals
- the model is 784-64-32-16-8-4-2-1-10
- will create assets/activation-1d-dist-hsic.jpg, produced from HSIC-Bot
- will create assets/activation-1d-dist-standard.jpg, produced from standard training
```sh
run_hsicbt -cfg config/niddle.yaml
```
- experiment sample (1d activation distribution. left:hsic; right:standard)
<img src="./assets/samples/activation-1d-dist-hsic.jpg"  width="256" height="256">
<img src="./assets/samples/activation-1d-dist-standard.jpg"  width="256" height="256">

#### Note
- format-train: training with single layer attached after the HSIC-trained network
- standard-train: the training with crossentorpy plus backpropagation