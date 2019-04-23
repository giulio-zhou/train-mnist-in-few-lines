import argparse
import keras
import numpy as np
from keras import optimizers
from keras.layers import Flatten, Reshape
from keras.layers import Input, Dense, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()

# Model definitions.
def standard_mlp():
    model = Sequential()
    model.add(Reshape((28*28,), input_shape=(28, 28, 1)))
    model.add(Dense(units=200))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    return model

def lenet():
    model = Sequential()
    model.add(Conv2D(20, 5, input_shape=(28, 28, 1)))
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
data, test_data = data.reshape(-1, 28, 28, 1), test_data.reshape(-1, 28, 28, 1)

# Preprocessing.
def train_fn(x):
    # Keras does not have an implementation of random cropping.
    def random_crop(img, random_crop_size):
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:y+dy, x:x+dx]
    x = 2.0 * x - 1.0
    x = np.pad(x, ((4, 4), (4, 4), (0, 0)), 'constant')
    x = random_crop(x, (28, 28))
    return x
def test_fn(x):
    return 2.0 * x - 1.0
train_datagen = ImageDataGenerator(preprocessing_function=train_fn).flow(
    data, labels, args.batch_size)
test_datagen = ImageDataGenerator(preprocessing_function=test_fn).flow(
    test_data, test_labels, args.batch_size)

# Optimizer.
if args.optimizer == 'sgd':
    optimizer = optimizers.SGD(lr=args.lr, momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = 'adam'

loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
model.fit_generator(train_datagen, epochs=args.num_epochs,
                    validation_data=test_datagen,
                    workers=args.num_workers, use_multiprocessing=True)
