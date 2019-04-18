import argparse
import keras
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
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
    model.add(tf.keras.layers.Reshape((28*28,), input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=200))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(units=10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def lenet():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(tf.keras.layers.Conv2D(20, 5))
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

# Dataset.
from utils import get_MNIST
data, labels, test_data, test_labels = get_MNIST()
data, test_data = data.reshape(-1, 28, 28), test_data.reshape(-1, 28, 28)
data, test_data = 2.0 * data - 1.0, 2.0 * test_data - 1.0

# Optimizer.
if args.optimizer == 'sgd':
    optimizer = tf.train.MomentumOptimizer(args.lr, 0.9)
elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=args.tpu_zone)
    )
)

def train_gen(batch_size):
    while True:
        offset = np.random.randint(0, data.shape[0] - batch_size)
        yield data[offset:offset+batch_size], labels[offset:offset+batch_size]

loss = 'categorical_crossentropy'
metrics = ['accuracy']
tpu_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
tpu_model.fit_generator(
    train_gen(args.batch_size),
    epochs=args.num_epochs,
    steps_per_epoch=int(np.ceil(len(data) / float(args.batch_size))),
    validation_data=(test_data, test_labels),
)
