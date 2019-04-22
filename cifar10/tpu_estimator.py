import argparse
import numpy as np
import os
import tensorflow as tf
from keras.datasets import cifar10
from resnet_model_tpu import resnet_v1
from tensorflow.contrib.tpu.python.tpu import async_checkpoint
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.core.protobuf import rewriter_config_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
parser.add_argument('--model_dir', required=True)
parser.add_argument('--num_shards', default=8, type=int)
parser.add_argument('--tpu_name', required=True)
parser.add_argument('--tpu_name_eval')
parser.add_argument('--tpu_zone', default='us-central1-f')
args = parser.parse_args()

# Model definitions.
def standard_mlp(features):
    input_layer = tf.reshape(features, [-1, 32*32*3])
    fc1 = tf.layers.dense(inputs=input_layer, units=200, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=fc1, units=10)
    return logits

# Network architecture.
if args.net == 'standard_mlp':
    net_fn = standard_mlp
elif args.net == 'lenet':
    net_fn = lenet
elif args.net == 'resnet50':
    network = resnet_v1(50, 10, 'channels_last')
    net_fn = lambda x: network(inputs=x, is_training=True)

# Dataset.
(data, labels), (test_data, test_labels) = cifar10.load_data()
data, test_data = 2.0 * (data / 255.) - 1.0, 2.0 * (test_data / 255.) - 1.0
data, test_data = data.astype(np.float32), test_data.astype(np.float32)
labels = labels[:, 0].astype(np.int32)
test_labels = test_labels[:, 0].astype(np.int32)
steps_per_epoch = (len(data) // args.batch_size) + \
                  int((len(data) % args.batch_size) > 0)
iterations_per_loop = args.eval_interval * steps_per_epoch

# Optimizer.
if args.optimizer == 'sgd':
    optimizer_fn = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)
elif args.optimizer == 'adam':
    optimizer_fn = lambda lr: tf.train.AdamOptimizer(lr)

# TPUEstimator functions.
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
    # (1.0 / 6, 0), (2.0 / 6, 1), (3.0 / 6, 2), (4.0 / 6, 3), (5.0 / 6, 4),
    # (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80), (0.0001, 90)
    (1.0, 0), (0.1, 60), (0.01, 120)
]

def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  scaled_lr = args.lr * (args.batch_size / 128.0)

  decay_rate = scaled_lr
  for mult, start_epoch in _LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch, decay_rate,
                          scaled_lr * mult)

  return decay_rate

def make_input_fn(data, labels):
    def input_fn(params):
        def data_aug(img, label):
            aug_img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
            aug_img = tf.random_crop(aug_img, [32, 32, 3])
            aug_img = tf.image.random_flip_left_right(aug_img)
            return aug_img, label
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(50000).repeat()
        dataset = dataset.map(data_aug)
        dataset = dataset.batch(params['batch_size'], drop_remainder=True)
        return dataset
    return input_fn
    
def model_fn(features, labels, mode, params):
    logits = net_fn(features)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    loss += params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

    # LR computation.
    global_step = tf.train.get_global_step()
    current_epoch = (
      tf.cast(global_step, tf.float32) / steps_per_epoch)
    # learning_rate = tf.train.exponential_decay(
    #     args.lr, global_step, 60 * steps_per_epoch, 0.1)
    learning_rate = learning_rate_schedule(current_epoch)
    
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = optimizer_fn(learning_rate)
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        # Metrics.
        # TODO: this is a hack to get the LR and epoch for Tensorboard.
        # Reimplement this when TPU training summaries are supported.
        lr_repeat = tf.reshape(
            tf.tile(tf.expand_dims(learning_rate, 0),
                    [params['batch_size'],]), [params['batch_size'], 1])
        ce_repeat = tf.reshape(
            tf.tile(tf.expand_dims(current_epoch, 0),
                    [params['batch_size'],]), [params['batch_size'], 1])
        def metric_fn(labels, logits, lr_repeat, ce_repeat):
            """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""
            predictions = tf.argmax(logits, axis=1)
            accuracy = tf.metrics.accuracy(labels, predictions)
            lr = tf.metrics.mean(lr_repeat)
            ce = tf.metrics.mean(ce_repeat)
            return {"accuracy": accuracy, "learning_rate": lr, "current_epoch": ce}
        eval_metrics = (metric_fn, [labels, logits, lr_repeat, ce_repeat])

    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss,
                                           train_op=train_op,
                                           eval_metrics=eval_metrics)

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
    # log_step_count_steps=FLAGS.log_step_count_steps,
    # session_config=tf.ConfigProto(
    #     graph_options=tf.GraphOptions(
    #         rewrite_options=rewriter_config_pb2.RewriterConfig(
    #             disable_meta_optimizer=True))),
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=steps_per_epoch,
        num_shards=args.num_shards,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
        .PER_HOST_V2))  # pylint: disable=line-too-long

params = dict(weight_decay=args.weight_decay)
tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    config=run_config,
    train_batch_size=args.batch_size,
    eval_batch_size=args.batch_size,
    params=params)

hooks = []
hooks.append(
    async_checkpoint.AsyncCheckpointSaverHook(
        checkpoint_dir=args.model_dir,
        save_steps=iterations_per_loop))

train_input_fn = make_input_fn(data, labels)
eval_input_fn = make_input_fn(test_data, test_labels)

if pid > 0:
    tpu_estimator.train(input_fn=train_input_fn,
                        steps=args.num_epochs * steps_per_epoch,
                        hooks=hooks)
else:
    for ckpt in evaluation.checkpoints_iterator(args.model_dir):
        eval_results = tpu_estimator.evaluate(
            input_fn=eval_input_fn,
            steps=len(test_data) // args.batch_size,
            checkpoint_path=ckpt)
        print("Eval results: %s" % eval_results)
