import argparse
import numpy as np
import tensorflow as tf

from utils import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
args = parser.parse_args()

def train(data, labels, sess, inputs, input_labels,
          outputs, loss, trainer, batch_size):
    train_loss = 0
    correct = 0
    total = 0
    indices = np.arange(len(data))
    np.random.shuffle(indices) 
    indices = np.hstack([indices, indices[:batch_size - (len(data) % batch_size)]])
    num_iters = len(indices) / batch_size
    for i in range(0, len(indices) - 1, batch_size):
        batch_idx = i // batch_size
        idx = indices[i:i+batch_size]
        X, y = data[idx], labels[idx]
        X = 2.0 * X - 1.0
        batch_loss, preds, _ = sess.run([loss, outputs, trainer],
                                         feed_dict={inputs: X, input_labels: y})

        train_loss += sum(batch_loss)
        predicted = np.argmax(preds, axis=1)
        true_label = np.argmax(y, axis=1)
        total += batch_size
        correct += np.sum(predicted == true_label)

        progress_bar(batch_idx, num_iters, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(data, labels, sess, inputs, input_labels,
         outputs, loss, batch_size):
    test_loss = 0
    correct = 0
    total = 0
    num_iters = int(np.ceil(float(len(data)) / batch_size))
    for i in range(0, len(data), batch_size):
        batch_idx = i // batch_size
        start, end = i, min(i + batch_size, len(data))
        X, y = data[start:end], labels[start:end]
        X = 2.0 * X - 1.0
        batch_loss, preds = sess.run([loss, outputs],
                                     feed_dict={inputs: X, input_labels: y})

        test_loss += sum(batch_loss)
        predicted = np.argmax(preds, axis=1)
        true_label = np.argmax(y, axis=1)
        total += end - start
        correct += np.sum(predicted == true_label)

        progress_bar(batch_idx, num_iters, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def standard_mlp(inputs):
    x = tf.reshape(inputs, [-1, 28 * 28])
    x = tf.layers.dense(x, 200, activation=tf.nn.relu)
    x = tf.layers.dense(x, 10)
    return x

inputs = tf.placeholder(tf.float32, [None, 28, 28])
input_labels = tf.placeholder(tf.float32, [None, 10])

# Network architecture.
if args.net == 'standard_mlp':
    outputs = standard_mlp(inputs)

# Optimizer.
if args.optimizer == 'sgd':
    optimizer = tf.train.MomentumOptimizer(args.lr, 0.9)
elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer()

# Dataset.
from utils import get_MNIST
data, labels, test_data, test_labels = get_MNIST()
data, test_data = data.reshape(-1, 28, 28), test_data.reshape(-1, 28, 28)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels,
                                                  logits=outputs)
trainer = optimizer.minimize(tf.reduce_mean(loss))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Train.
for epoch in range(args.num_epochs):
    print('Epoch %d' % epoch)
    train(data, labels, sess, inputs, input_labels,
          outputs, loss, trainer, args.batch_size)
    test(test_data, test_labels, sess, inputs, input_labels,
         outputs, loss, args.batch_size)
