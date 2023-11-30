import torch
import torch.nn.functional as F
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
from nn_utils import Net, DEVICE, TRAINLOADER, train_nn, test_nn, freeze_parameters

torch.cuda.empty_cache()

# create a new neural network
torch.manual_seed(1)
net = Net()
net.to(DEVICE)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.0)

train_nn(net=net, epochs=15, optimizer=optimizer)
test_nn(net=net, verbose=True)

PATH = "./nn-models/cifar10-nn-model"
torch.save(net.state_dict(), PATH)