import argparse
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from resnet_model_tpu import resnet_v1
from tensorflow.core.protobuf import rewriter_config_pb2

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--eval_interval', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net', default='standard_mlp')
parser.add_argument('--model_dir', required=True)
parser.add_argument('--tpu_name', required=True)
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

# Optimizer.
if args.optimizer == 'sgd':
    optimizer_fn = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)
elif args.optimizer == 'adam':
    optimizer_fn = lambda lr: tf.train.AdamOptimizer(lr)

# TPUEstimator functions.
def make_input_fn(data, labels, batch_size):
    def input_fn(params):
        def data_aug(img, label):
            aug_img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
            aug_img = tf.random_crop(img, [32, 32, 3])
            aug_img = tf.image.random_flip_left_right(aug_img)
            return aug_img, label
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.shuffle(50000).repeat()
        dataset = dataset.map(data_aug)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset
    return input_fn
    
def model_fn(features, labels, mode, params):
    logits = net_fn(features)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    
    learning_rate = tf.train.exponential_decay(
        args.lr, tf.train.get_global_step(), 40*steps_per_epoch, 0.1)
    
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = optimizer_fn(learning_rate)
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        # Metrics.
        global_step = tf.train.get_global_step()
        current_epoch = (
          tf.cast(global_step, tf.float32) / steps_per_epoch)
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

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    args.tpu_name, zone=args.tpu_zone)

run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=args.model_dir,
    # model_dir=FLAGS.model_dir,
    # save_checkpoints_steps=save_checkpoints_steps,
    # log_step_count_steps=FLAGS.log_step_count_steps,
    # session_config=tf.ConfigProto(
    #     graph_options=tf.GraphOptions(
    #         rewrite_options=rewriter_config_pb2.RewriterConfig(
    #             disable_meta_optimizer=True))),
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=steps_per_epoch,
        num_shards=8))
        # per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
        # .PER_HOST_V2))  # pylint: disable=line-too-long

tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    train_batch_size=args.batch_size,
    eval_batch_size=args.batch_size,
    config=run_config)

train_input_fn = make_input_fn(data, labels, args.batch_size)
eval_input_fn = make_input_fn(test_data, test_labels, args.batch_size)
for i in range(0, args.num_epochs, args.eval_interval):
    print('train')
    tpu_estimator.train(input_fn=train_input_fn,
                        steps=steps_per_epoch * args.eval_interval)
    print('eval')
    eval_results = tpu_estimator.evaluate(
        input_fn=eval_input_fn, steps=len(test_data) // args.batch_size)
    print("Eval results: %s" % eval_results)
