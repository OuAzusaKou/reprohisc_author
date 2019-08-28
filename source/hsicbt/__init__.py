import 	torch
from   	torch import nn, optim
from   	torch.autograd import Variable
import 	torch.nn.functional as F
from 	torch.utils.data import DataLoader
from 	torchvision import datasets, transforms

torch.manual_seed(1234)

from tqdm import tqdm
import numpy as np
import os

if not os.path.exists("./models"):
	os.makedirs("./models")
if not os.path.exists("./assets"):
	os.makedirs("./assets")
