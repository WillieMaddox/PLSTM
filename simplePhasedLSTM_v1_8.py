from __future__ import division
from datetime import datetime
import tensorflow as tf
import numpy as np
from tabulate import tabulate
from tensorflow.contrib.rnn import LSTMStateTuple
# from tensorflow.contrib.rnn import PhasedLSTMCell
from PhasedLSTMCell_v1_8 import PhasedLSTMCell

# Unit test for Phased LSTM
# Here I implement the first task described in the original paper of PLSTM
#   https://arxiv.org/abs/1610.09513
# which is the sine waves discrimination

flags = tf.flags
flags.DEFINE_boolean("async", True, "Use asynchronous sampling")
flags.DEFINE_float("resolution", 0.1, "Sampling resolution if async is set to False")
flags.DEFINE_integer("n_hidden", 100, "hidden units in the recurrent layer")
flags.DEFINE_integer("n_epochs", 30, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("b_per_epoch", 80, "batches per epoch")
flags.DEFINE_integer("n_layers", 2, "hidden units in the recurrent layer")
flags.DEFINE_integer("max_length", 200, "max length of sin waves")
flags.DEFINE_integer("min_length", 150, "min length of sine waves")
flags.DEFINE_float("min_f_on", 35, "min frequency for the on set")
flags.DEFINE_float("max_f_on", 65, "max frequency for the on set")
flags.DEFINE_float("min_f_off", 1, "min frequency for the off set")
flags.DEFINE_float("max_f_off", 100, "max frequency for the off set")
flags.DEFINE_float("exp_init", 3., "Value for initialization of Tau")
FLAGS = flags.FLAGS

# Net Params
n_input = 1
n_out = 2
tpe = "async" if FLAGS.async else "sync"


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M_%S'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


run_name = '{}_{}_res_{}_hid_{}_exp_{}'.format(get_datetime_now(), tpe, FLAGS.resolution, FLAGS.n_hidden, FLAGS.exp_init)


def gen_async_sin(async_sampling, resolution=None, batch_size=32, on_target_period=(5, 6), off_target_period=(1, 100), max_len=125, min_len=85):

    half_batch = int(batch_size / 2)
    full_length = off_target_period[1] - on_target_period[1] + on_target_period[0] - off_target_period[0]

    # generate random periods
    posTs = np.random.uniform(on_target_period[0], on_target_period[1], half_batch)
    size_low = np.floor((on_target_period[0] - off_target_period[0]) * half_batch / full_length).astype('int32')
    size_high = np.ceil((off_target_period[1] - on_target_period[1]) * half_batch / full_length).astype('int32')
    low_vec = np.random.uniform(off_target_period[0], on_target_period[0], size_low)
    high_vec = np.random.uniform(on_target_period[1], off_target_period[1], size_high)
    negTs = np.hstack([low_vec, high_vec])

    # generate random lengths
    if async_sampling:
        lens = np.random.uniform(min_len, max_len, batch_size)
    else:
        max_len *= int(1 / resolution)
        min_len *= int(1 / resolution)
        lens = np.random.uniform(min_len, max_len, batch_size)

    # generate random number of samples
    if async_sampling:
        samples = np.random.uniform(min_len, max_len, batch_size).astype('int32')
    else:
        samples = lens

    start_times = np.array([np.random.uniform(0, max_len - duration) for duration in lens])
    x = np.zeros((batch_size, max_len, 1))
    y = np.zeros((batch_size, 2))
    t = np.zeros((batch_size, max_len, 1))
    for i, s, l, n in zip(range(batch_size), start_times, lens, samples):
        if async_sampling:
            time_points = np.reshape(np.sort(np.random.uniform(s, s + l, n)), [-1, 1])
        else:
            time_points = np.reshape(np.arange(s, s + n * resolution, step=resolution), [-1, 1])

        if i < half_batch:  # positive
            _tmp_x = np.squeeze(np.sin(time_points * 2 * np.pi / posTs[i]))
            x[i, :len(_tmp_x), 0] = _tmp_x
            t[i, :len(_tmp_x), 0] = np.squeeze(time_points)
            y[i, 0] = 1.
        else:
            _tmp_x = np.squeeze(np.sin(time_points * 2 * np.pi / negTs[i - half_batch]))
            x[i, :len(_tmp_x), 0] = _tmp_x
            t[i, :len(_tmp_x), 0] = np.squeeze(time_points)
            y[i, 1] = 1.

    x = np.squeeze(np.stack([x, t], 2))

    return x, y, samples


def RNN(_X, _weights, _biases, lens, initial_states):

    cells = [PhasedLSTMCell(FLAGS.n_hidden, use_peepholes=True) for _ in range(FLAGS.n_layers)]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, _X,
                                       sequence_length=lens,
                                       initial_state=tuple(initial_states),
                                       dtype=tf.float32)

    outputs = tf.slice(outputs, [0, 0, 0], [-1, -1, FLAGS.n_hidden])

    # TODO better (?) in lack of smart indexing
    batch_size = tf.shape(outputs)[0]
    max_len = tf.shape(outputs)[1]
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_len + (lens - 1)
    flat = tf.reshape(outputs, [-1, out_size])
    relevant = tf.gather(flat, index)

    return tf.nn.bias_add(tf.matmul(relevant, _weights['out']), _biases['out'])


def main(_):
    # inputs (plus 1 for the time)
    x = tf.placeholder(tf.float32, [None, None, n_input + 1])

    # length of the samples -> for dynamic_rnn
    lens = tf.placeholder(tf.int32, [None])

    # labels
    y = tf.placeholder(tf.float32, [None, 2])

    # weights from input to hidden
    weights = {
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden, n_out], dtype=tf.float32))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([n_out], dtype=tf.float32))
    }

    # Register weights to be monitored by tensorboard
    w_out_hist = tf.summary.histogram("weights_out", weights['out'])
    b_out_hist = tf.summary.histogram("biases_out", biases['out'])

    # Let's define the training and testing operations
    print("Compiling RNN...", )
    c0 = tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32)
    h0 = tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32)
    initial_states = [LSTMStateTuple(c0, h0)] * FLAGS.n_layers
    # initial_states = [LSTMStateTuple(c0, h0) for _ in range(FLAGS.n_layers)]
    # initial_states = [LSTMStateTuple(tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32),
    #                                  tf.zeros([FLAGS.batch_size, FLAGS.n_hidden], tf.float32)) for _
    #                   in range(FLAGS.n_layers)]
    predictions = RNN(x, weights, biases, lens, initial_states)
    print("DONE!")

    # Register initial_state to be monitored by tensorboard
    initial_state_hist = tf.summary.histogram("initial_states", initial_states)

    print("Compiling cost functions...", )
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
    print("DONE!")

    # I like to log the gradients
    tvars = tf.trainable_variables()
    grads = tf.gradients(cost, tvars)

    grads_hist = [tf.summary.histogram("grads_{}".format(i), k) for i, k in enumerate(grads) if k is not None]
    merged_grads = tf.summary.merge([grads_hist] + [w_out_hist, b_out_hist] + [initial_state_hist])
    # merged_grads = tf.summary.merge([grads_hist] + [w_out_hist, b_out_hist])
    cost_summary = tf.summary.scalar("cost", cost)
    cost_val_summary = tf.summary.scalar("cost_val", cost)

    print("Calculating gradients...", )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print("DONE!")

    # evaluation
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    accuracy_val_summary = tf.summary.scalar("accuracy_val", accuracy)

    # run the model
    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        print("Initializing variables...", )
        sess.run(init)
        print("DONE!")

        writer = tf.summary.FileWriter("phasedLSTM_run/{}".format(run_name), sess.graph)

        # training loop
        for step in range(FLAGS.n_epochs):
            train_cost = 0
            train_acc = 0
            for i in range(FLAGS.b_per_epoch):

                batch_xs, batch_ys, leng = gen_async_sin(
                    FLAGS.async,
                    FLAGS.resolution,
                    FLAGS.batch_size,
                    [FLAGS.min_f_on, FLAGS.max_f_on],
                    [FLAGS.min_f_off, FLAGS.max_f_off],
                    FLAGS.max_length,
                    FLAGS.min_length
                )

                res = sess.run(
                    [optimizer, cost, accuracy, grads, cost_summary, accuracy_summary, merged_grads],
                    feed_dict={
                        x: batch_xs,
                        y: batch_ys,
                        lens: leng}
                )

                writer.add_summary(res[6], step * FLAGS.b_per_epoch + i)
                writer.add_summary(res[4], step * FLAGS.b_per_epoch + i)
                writer.add_summary(res[5], step * FLAGS.b_per_epoch + i)
                train_cost += res[1] / FLAGS.b_per_epoch
                train_acc += res[2] / FLAGS.b_per_epoch

            # test accuracy
            # wipe initial_states before testing
            for i, _ in enumerate(initial_states):
                initial_states[i] = None

            test_xs, test_ys, leng = gen_async_sin(
                FLAGS.async,
                FLAGS.resolution,
                FLAGS.batch_size,
                [FLAGS.min_f_on, FLAGS.max_f_on],
                [FLAGS.min_f_off, FLAGS.max_f_off],
                FLAGS.max_length,
                FLAGS.min_length
            )

            loss_test, acc_test, summ_cost, summ_acc = sess.run(
                [cost, accuracy, cost_val_summary, accuracy_val_summary],
                feed_dict={
                    x: test_xs,
                    y: test_ys,
                    lens: leng}
            )

            writer.add_summary(summ_cost, step * FLAGS.b_per_epoch + i)
            writer.add_summary(summ_acc, step * FLAGS.b_per_epoch + i)
            table = [["Train", train_cost, train_acc], ["Test", loss_test, acc_test]]
            headers = ["Epoch={}".format(step), "Cost", "Accuracy"]
            print(tabulate(table, headers, tablefmt='grid'))

            # wipe initial_states after testing
            for i, _ in enumerate(initial_states):
                initial_states[i] = None


if __name__ == "__main__":
    tf.app.run()
