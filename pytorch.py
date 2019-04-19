import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from utils import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()

def train(net, optimizer, criterion, device, dataset_loader):
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataset_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net.forward(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(net, criterion, device, dataset_loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataset_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Model definitions.
class StandardMLP(nn.Module):
    def __init__(self):
        super(StandardMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Network architecture.
if args.net == 'standard_mlp':
    net = StandardMLP()
elif args.net == 'lenet':
    net = LeNet()
net.to(args.device)

if args.device == 'cuda':
    net = torch.nn.DataParallel(net)

# Optimizer.
if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters())

# Preprocessing.
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Dataset.
trainset = torchvision.datasets.MNIST('pytorch_mnist', train=True,
                                      transform=transform_train, download=True)
testset = torchvision.datasets.MNIST('pytorch_mnist', train=False,
                                     transform=transform_test, download=True)
    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                         num_workers=args.num_workers)
# Train.
criterion = nn.CrossEntropyLoss()
for epoch in range(args.num_epochs):
    print('Epoch %d' % epoch)
    train(net, optimizer, criterion, args.device, trainloader)
    test(net, criterion, args.device, testloader)
