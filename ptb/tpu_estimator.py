import argparse
import numpy as np
import os
import tensorflow as tf
import time
from keras.preprocessing.text import Tokenizer
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--num_steps', default=20, type=int)
parser.add_argument('--vocab_size', default=10000, type=int)
parser.add_argument('--char_mode', action='store_true')

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--net', default='small')

parser.add_argument('--use_tpu', action='store_true')
parser.add_argument('--model_dir', required=True)
parser.add_argument('--num_shards', default=8, type=int)
parser.add_argument('--tpu_name')
parser.add_argument('--tpu_name_eval')
parser.add_argument('--tpu_zone', default='us-central1-f')
args = parser.parse_args()

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().replace('\n', '<eos>')

def segment(data):
    num_batches = len(data) // args.num_steps
    data = np.array(data[:num_batches * args.num_steps])
    data = data.reshape(num_batches, args.num_steps)
    return data
    
# Dataset.
tokenizer = Tokenizer(num_words=args.vocab_size)
if args.char_mode:
    file_template = '%s/ptb.char.%s.txt'
else:
    file_template = '%s/ptb.%s.txt'
train, val, test = [read_file(file_template % (args.data_dir, x))
                    for x in ['train', 'valid', 'test']]
tokenizer = Tokenizer(num_words=args.vocab_size)
tokenizer.fit_on_texts([train, val, test])
train_enc, val_enc, test_enc = tokenizer.texts_to_sequences([train, val, test])
train_len, val_len = len(segment(train_enc)), len(segment(val_enc))

steps_per_epoch = (train_len // args.batch_size) + \
                  int((train_len % args.batch_size) > 0)
iterations_per_loop = args.eval_interval * steps_per_epoch
print(train_len, val_len, steps_per_epoch, iterations_per_loop)

# Model definitions.
def small(features, initial_state=None):
    batch_size = tf.shape(features)[0]
    all_logits = []
    lstm = tf.contrib.rnn.BasicLSTMCell(200)
    embedding = tf.get_variable(
        "embedding", [args.vocab_size, 200], dtype=tf.float32)
    softmax_w = tf.layers.Dense(args.vocab_size)
    if initial_state is None:
        initial_state = state = lstm.zero_state(batch_size, dtype=tf.float32)
    for i in range(args.num_steps):
        inputs = tf.nn.embedding_lookup(embedding, features[:, i])
        output, state = lstm(inputs, state)
        logits = softmax_w(output)
        all_logits.append(logits)
    all_logits = tf.stack(all_logits, axis=1)
    return all_logits, state

# Network architecture.
if args.net == 'small':
    net_fn = small

# Optimizer.
if args.optimizer == 'sgd':
    optimizer_fn = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)

# TPUEstimator functions.
def make_input_fn(data):
    def input_fn(params):
        pred_data = segment(data[:-1])
        target_data = segment(data[1:])
        num_batches = min(len(pred_data), len(target_data))
        pred_data = pred_data[:num_batches]
        target_data = target_data[:num_batches]
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(pred_data),
            tf.data.Dataset.from_tensor_slices(target_data)
        ))
        dataset = dataset.map(lambda x, y: {'source': x, 'target': y})
        dataset = dataset.shuffle(num_batches).repeat()
        dataset = dataset.batch(params['batch_size'], drop_remainder=True)
        return dataset
    return input_fn

def train_fn(source, target):
    logits, lstm_state = net_fn(source)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logits))
    # No learning rate decay right now.
    optimizer = optimizer_fn(args.lr)
    if args.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op
    )

def eval_fn(source, target):
    logits, lstm_state = net_fn(source)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logits))
    def metric_fn(labels, logits):
        labels = tf.cast(labels, tf.int64)
        return {
            'recall@1': tf.metrics.recall_at_k(labels, logits, 1),
            'recall@5': tf.metrics.recall_at_k(labels, logits, 5)
        }
    eval_metrics = (metric_fn, [target, logits])
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metrics=eval_metrics
    )

def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return train_fn(features['source'], features['target'])
    elif mode == tf.estimator.ModeKeys.EVAL:
        return eval_fn(features['source'], features['target'])

if args.use_tpu:
    pid = os.fork()
    if pid > 0 or (args.tpu_name_eval is None):
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name, zone=args.tpu_zone)
    else:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            args.tpu_name_eval, zone=args.tpu_zone)
    print(pid, tpu_cluster_resolver.get_master())
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=args.model_dir,
        save_checkpoints_steps=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=steps_per_epoch,
            num_shards=args.num_shards,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
            .PER_HOST_V2))  # pylint: disable=line-too-long
else:
    run_config = tf.contrib.tpu.RunConfig(
        model_dir=args.model_dir,
        save_checkpoints_steps=None)

tpu_estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=args.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=args.batch_size,
    eval_batch_size=args.batch_size)

hooks = []
hooks.append(
    async_checkpoint.AsyncCheckpointSaverHook(
        checkpoint_dir=args.model_dir,
        save_steps=iterations_per_loop))

# Train/eval.
train_input_fn = make_input_fn(train_enc)
eval_input_fn = make_input_fn(val_enc)
if args.use_tpu:
    if pid > 0:
        tpu_estimator.train(input_fn=train_input_fn,
                            steps=args.num_epochs * steps_per_epoch,
                            hooks=hooks)
        # Sleep so that eval can finish before closing.
        time.sleep(360)
    else:
        for ckpt in evaluation.checkpoints_iterator(args.model_dir):
            eval_results = tpu_estimator.evaluate(
                input_fn=eval_input_fn,
                steps=val_len // args.batch_size,
                checkpoint_path=ckpt)
            print("Eval results: %s" % eval_results)
else:
    for i in range(0, args.num_epochs, args.eval_interval):
        tpu_estimator.train(input_fn=train_input_fn,
                            steps=args.eval_interval * steps_per_epoch)
        eval_results = tpu_estimator.evaluate(
            input_fn=eval_input_fn,
            steps=val_len // args.batch_size)
        print("Eval results: %s" % eval_results)
