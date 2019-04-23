import argparse
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from resnet_model import ResNet50

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
parser.add_argument('--tpu_name', required=True)
parser.add_argument('--tpu_zone', default='us-central1-f')
args = parser.parse_args()

# Model definitions.
def standard_mlp():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((32*32*3,), input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Dense(units=200))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(units=10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def lenet():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(20, 5, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(50, 5))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=500))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(units=10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

# Network architecture.
if args.net == 'standard_mlp':
    model = standard_mlp()
elif args.net == 'lenet':
    model = lenet()
elif args.net == 'resnet50':
    model = ResNet50(10)

# Dataset.
(data, labels), (test_data, test_labels) = cifar10.load_data()
data, test_data = 2.0 * (data / 255.) - 1.0, 2.0 * (test_data / 255.) - 1.0

# Optimizer.
if args.optimizer == 'sgd':
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(args.lr)

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
    x = random_crop(x, (32, 32))
    return x
def test_fn(x):
    return 2.0 * x - 1.0
train_datagen = ImageDataGenerator(preprocessing_function=train_fn, horizontal_flip=True).flow(
    data, labels, args.batch_size)
test_datagen = ImageDataGenerator(preprocessing_function=test_fn).flow(
    test_data, test_labels, args.batch_size)

loss = tf.keras.losses.sparse_categorical_crossentropy
metrics = ['sparse_categorical_accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=args.tpu_zone)
    )
)
tpu_model.fit_generator(
    train_datagen,
    epochs=args.num_epochs,
    validation_data=test_datagen,
    workers=8, use_multiprocessing=True,
)
