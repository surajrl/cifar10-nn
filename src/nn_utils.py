import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

MINI_BATCH_SIZE = 128

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
    )

torch.cuda.empty_cache()

'''
    Load the Dataset
'''
TRAINSET = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
    )

TRAINLOADER = torch.utils.data.DataLoader(
    TRAINSET,
    batch_size=MINI_BATCH_SIZE,
    shuffle=True, # reshuffle data at every epoch
    num_workers=2
    )

TESTSET = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
    )

TESTLOADER = torch.utils.data.DataLoader(
    TESTSET,
    batch_size=MINI_BATCH_SIZE,
    shuffle=False,
    num_workers=2
    )

'''
    Define the Neural Network Architecture
'''
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels), 
        nn.ReLU(inplace=True)
        ]
    
    if pool:
        layers.append(nn.MaxPool2d(2))
    
    return nn.Sequential(*layers)  

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)

        # residual block start
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        # residual block end

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.maxPool3 = nn.MaxPool2d(kernel_size=2)

        # residual block start
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchNorm7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batchNorm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        # residual block end        
        
        self.out = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.res1(x) + x
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.res2(x) + x

        # first block
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)

        # second block
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # residual block
        x_res1 = F.max_pool2d(x, kernel_size=2) # save for residual block
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        x = x + x_res1
        
        # third block
        x = self.conv5(x)
        x = self.batchNorm5(x)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2)

        x = self.conv6(x)
        x = self.batchNorm6(x)
        x = F.relu(x)
        x = F.max_pool2d(kernel_size=2)

        # residual block
        x = self.conv7(x)
        x = self.batchNorm7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.batchNorm8(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, kernel_size=4)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = self.out(x)

        return x

'''
    Train the Neural Network using Backpropagation with Cross Entropy as the Loss Function
'''
def train_nn(net: nn.Module, epochs: int, optimizer: torch.optim.Optimizer):    
    print(f"Using {DEVICE} device")
    print(f'Initialising training ...')
    print(f'- Epochs: {epochs}')
    print(f'- Mini batch size: {MINI_BATCH_SIZE}')
    print(f'- Optimiser: {optimizer}')
    print(f'- Loss function: {F.cross_entropy.__name__}')

    accuracy_track = []
    loss_track = []

    # loop over the dataset multiple times
    for epoch in range(epochs):
        # loop over the dataset by mini-batch
        for mini_batch in TRAINLOADER:
            images = mini_batch[0].to(DEVICE)
            labels = mini_batch[1].to(DEVICE)

            optimizer.zero_grad() # zero the parameter gradients

            preds = net(images) # forward mini-batch

            loss = F.cross_entropy(preds, labels) # calculate loss
            loss.backward() # calculate gradients with respect to each weight
            optimizer.step() # update weights

            accuracy = test_nn(net=net, verbose=False)
            
            # track
            accuracy_track.append(accuracy)
            loss_track.append(loss.item())

        print(f'\nEpoch {epoch} finished')

    # plot
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Evaluations of the loss function')
    
    ax1.set_ylabel('Accuracy', color='b')
    ax1.plot(np.array(accuracy_track), color='b', marker='.', label='Accuracy', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='r')
    ax2.plot(np.array(loss_track), color='r', marker='.', label='Loss', linewidth=0.5)

    fig.tight_layout()
    fig.legend()
    plt.show()

'''
    Test the Neural Network
'''
def test_nn(net: nn.Module, verbose: bool):
    # test the neural network
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in TESTLOADER:
            images = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct // total

    if verbose:
        print('Testing on 10,000 test images ...')
        print(f'- Correct: {correct}')
        print(f'- Total: {total}')
        print(f'- Accuracy: {accuracy}')

    return accuracy

'''
    Freeze All the Parameters Except the Last Layer and Randomize Last Layer
'''
def freeze_parameters(net: nn.Module):
    # freeze all the parameters in the neural network
    for param in net.parameters():
        param.requires_grad = False

    # unfreeze all the parameters from the last layer and randomise the weights
    for param in net.out.parameters():
        param.requires_grad = True
        param.data = torch.rand(param.size(), device=DEVICE)