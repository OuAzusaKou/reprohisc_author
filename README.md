# HSIC-Bottleneck
This is the released repo for our work entitled `The HSIC Bottleneck: Deep Learning without Back-Propagation`. All the experiments were produced by this repository.

### Environment
- pytorch-1.1.0
- torchvision-0.3.0
- numpy-1.16.4
- scipy-1.3.0
- tqdm-4.33.0
- yaml-5.1.1

### Usage

Each run is according to the specific task generating figure in the paper. To reproduce all the experiments that we have in the paper, you could run our batch script by the following instruction:
```sh
# in bash
git clone git@gitlab.com:gladiator8072/hsic-bottleneck.git 
source env.sh
batch.sh
```


Please visit the running procedure page ([link](config/README.md)) for more information.

### Cite

Please cite our work if it is relevant to your research work, thanks!

```
@article{Ma2019TheHB,
  title={The HSIC Bottleneck: Deep Learning without Back-Propagation},
  author={Wan-Duo Ma and J. P. Lewis and W. Bastiaan Kleijn},
  journal={ArXiv},
  year={2019},
  volume={abs/1908.01580}
}
```