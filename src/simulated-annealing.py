import torch
import torch.nn.functional as F
import numpy as np
import random  
import matplotlib.pyplot as plt 
import nn_utils

try:
    PATH = './nn-models/cifar10-nn-model'
    # load the pretrained NN model
    net = nn_utils.Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device=nn_utils.DEVICE)
except FileNotFoundError:
    PATH = './src/nn-models/cifar10-nn-model'
    # load the pretrained NN model
    net = nn_utils.Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device=nn_utils.DEVICE)


MIN_BOUND = -1.0
MAX_BOUND = 1.0
N_DIMENSIONS = sum(param.numel() for param in net.out.parameters())
TEMP = 100
STEP_SIZE = 1.0

def objective_function(x):
    # put parameters into the neural network
    x = torch.as_tensor(data=x, device=nn_utils.DEVICE, dtype=torch.float32)
    net.out.weight = torch.nn.Parameter(data=x[0:640].reshape(10, 64))
    net.out.bias = torch.nn.Parameter(data=x[640:650])

    # calculate loss
    with torch.no_grad():
        running_loss = 0
        for mini_batch in nn_utils.TRAINLOADER:
            images = mini_batch[0].to(nn_utils.DEVICE)
            labels = mini_batch[1].to(nn_utils.DEVICE)
            
            pred = net(images)
            loss = F.cross_entropy(pred, labels)
            running_loss += loss.item()

    return running_loss

# generate a random solution
x_cb = []
for i in range(N_DIMENSIONS):
    x_cb.append(random.uniform(MIN_BOUND, MAX_BOUND))

objective_functions = []

for i in range(50):
    # calculate temperature for current epoch
    t = TEMP / float(i + 1)
    # perturb the current solution
    x_per = x_cb + np.random.uniform(MIN_BOUND, MAX_BOUND, size=N_DIMENSIONS).astype(float) * STEP_SIZE
    # calculate fitness value of perturb solution and compare with current solution
    J_x_per = objective_function(x_per)
    J_x_cb = objective_function(x_cb)
    if J_x_per < J_x_cb:
        x_cb, J_x_cb = x_per, J_x_per
        objective_functions.append(J_x_cb)
    else:
        diff = J_x_per - J_x_cb
        if np.random.rand() < np.exp(-diff / t):
            # accept worst solution
            x_cb, J_x_cb = x_per, x_cb
            objective_functions.append(J_x_cb)
    
    x = torch.as_tensor(data=x_cb, device=nn_utils.DEVICE, dtype=torch.float32)
    net.out.weight = torch.nn.Parameter(data=x[0:640].reshape(10, 64))
    net.out.bias = torch.nn.Parameter(data=x[640:650])
    print(f'Iteration {i} -- Error: {J_x_cb} -- Accuracy: {nn_utils.test_nn(net=net, verbose=False)}')

# put the final parameters in the neural network
x_cb = torch.as_tensor(data=x_cb, device=nn_utils.DEVICE, dtype=torch.float32)
net.out.weight = torch.nn.Parameter(data=x_cb[0:640].reshape(10, 64))
net.out.bias = torch.nn.Parameter(data=x_cb[640:650])

# test
nn_utils.test_nn(net=net, verbose=True)

# plot
fig, ax = plt.subplots()
ax.set_xlabel('Iterations')
ax.set_ylabel('Objective function')
ax.plot(np.array(objective_functions), color='b', marker='.', label='Objective function', linestyle='dashed', linewidth=0.5)
fig.legend()
plt.show()
