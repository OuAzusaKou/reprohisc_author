import  torch
from    torch import nn, optim
from    torch.autograd import Variable
import  torch.nn.functional as F
from    torch.utils.data import DataLoader
from    torchvision import datasets, transforms

torch.manual_seed(1234)


from tqdm import tqdm
import numpy as np
import yaml
import scipy as sp
import os

if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./assets"):
    os.makedirs("./assets")
if not os.path.exists("./assets/data"):
    os.makedirs("./assets/data")
if not os.path.exists("./assets/logs"):
    os.makedirs("./assets/logs")
if not os.path.exists("./assets/exp"):
    os.makedirs("./assets/exp")
if not os.path.exists("./assets/tmp"):
    os.makedirs("./assets/tmp")