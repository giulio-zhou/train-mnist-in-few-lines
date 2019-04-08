import argparse
import keras
import numpy as np
from keras import optimizers
from keras.layers import Flatten, Reshape
from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.models import Model, Sequential

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
args = parser.parse_args()

def train(model, data, labels, batch_size):
    model.fit(data, labels, epochs=1, batch_size=batch_size)

def test(model, data, labels, batch_size):
    print(model.evaluate(data, labels, batch_size=batch_size))

# Model definitions.
def standard_mlp():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(units=200, input_dim=28*28))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    return model

def lenet():
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Conv2D(20, 5))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(50, 5))
    model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(units=500))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    return model

# Network architecture.
if args.net == 'standard_mlp':
    model = standard_mlp()
elif args.net == 'lenet':
    model = lenet()

# Dataset.
from utils import get_MNIST
data, labels, test_data, test_labels = get_MNIST()
data, test_data = data.reshape(-1, 28, 28), test_data.reshape(-1, 28, 28)
data, test_data = 2.0 * data - 1.0, 2.0 * test_data - 1.0

# Optimizer.
if args.optimizer == 'sgd':
    optimizer = optimizers.SGD(lr=args.lr, momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = 'adam'

loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
for epoch in range(args.num_epochs):
    print('Epoch %d' % epoch)
    train(model, data, labels, args.batch_size)
    test(model, test_data, test_labels, args.batch_size)
