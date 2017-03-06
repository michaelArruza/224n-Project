#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3: Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

#import matplotlib
#import matplotlib.pyplot as plt

from util import Progbar, minibatches
from model import Model

from q3_gru_cell import GRUCell
from q2_rnn_cell import RNNCell

#from tensorflow.models.rnn import seq2seq

#matplotlib.use('TkAgg')
logger = logging.getLogger("hw3.q3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    max_length = 15 # Length of sequence used.
    batch_size = 100
    n_epochs = 40
    lr = 0.2
    max_grad_norm = 5.
    num_encoder_symbols = 100
    num_decoder_symbols = 100

    def pad_sequences(data, max_length):
        """Ensures each input-output seqeunce pair in @data is of length
        @max_length by padding it with zeros and truncating the rest of the
        sequence.

        TODO: In the code below, for every sentence, labels pair in @data,
        (a) create a new sentence which appends zero feature vectors until
        the sentence is of length @max_length. If the sentence is longer
        than @max_length, simply truncate the sentence to be @max_length
        long.
        (b) create a new label sequence similarly.
        (c) create a _masking_ sequence that has a True wherever there was a
        token in the original sequence, and a False for every padded input.

        Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
        0, 0], and max_length = 5, we would construct
            - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
            - a new label seqeunce: [1, 0, 0, 4, 4], and
            - a masking seqeunce: [True, True, True, False, False].

        Args:
            data: is a list of (sentence, labels) tuples. @sentence is a list
                containing the words in the sentence and @label is a list of
                output labels. Each word is itself a list of
                @n_features features. For example, the sentence "Chris
                Manning is amazing" and labels "PER PER O O" would become
                ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
                the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
                is the list of labels.
            max_length: the desired length for all input/output sequences.
        Returns:
            a new list of data points of the structure (sentence', labels', mask).
            Each of sentence', labels' and mask are of length @max_length.
            See the example above for more details.
        """
        ret = []

        # Use this zero vector when padding sequences.
        zero_vector = [0] * Config.n_features
        zero_label = 4 # corresponds to the 'O' tag

        for sentence, labels in data:
            ### YOUR CODE HERE (~4-6 lines)
            sentence_ = sentence[:]
            labels_ = labels[:]
            seq_length = min(len(sentence), max_length)
            sentence_ = sentence[:seq_length] + [zero_vector] * max(max_length - len(sentence), 0)
            labels_ = labels[:seq_length] + [zero_label] * max(max_length - len(sentence), 0)
            mask = [True for x in range(seq_length)] + [False] * max(max_length - len(sentence), 0)
            ret.append((sentence_, labels_, mask))
            ### END YOUR CODE ###
        return ret

class SequencePredictor(Model):

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        allEmbeddings = tf.Variable(self.pretrained_embeddings)
        selectedEmbeddings = tf.nn.embedding_lookup(allEmbeddings, self.input_placeholder)
        #self.max_length -> self.config.max_length
        embeddings = tf.reshape(selectedEmbeddings, (-1, self.config.max_length, self.config.n_features * self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        NOTE: You do not have to do anything here.
        """
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(30, 300, 1), name="x")
        self.labels_placeholder = tf.placeholder(tf.float32, shape=(30, 1), name="y")
        #self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_prediction_op(self):
        """Runs an rnn on the input using TensorFlows's
        @tf.nn.dynamic_rnn function, and returns the final state as a prediction.

        TODO:
            - Call tf.nn.dynamic_rnn using @cell below. See:
              https://www.tensorflow.org/api_docs/python/nn/recurrent_neural_networks
            - Apply a sigmoid transformation on the final state to
              normalize the inputs between 0 and 1.

        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        # Pick out the cell to use here.
        if self.config.cell == "rnn":
            cell = RNNCell(1, 1)
        elif self.config.cell == "gru":
            cell = GRUCell(1, 1)
        elif self.config.cell == "lstm":
            cell = tf.nn.rnn_cell.LSTMCell(1)
        else:
            raise ValueError("Unsupported cell type.")

        x = self.inputs_placeholder
        y = self.labels_placeholder
        ### YOUR CODE HERE (~2-3 lines)
        # outputs, state = seq2seq.embedding_rnn_seq2seq(x, y, cell, self.config.num_encoder_symbols,
        #                 self.config.num_decoder_symbols, embedding_size)
        # outputs, state = tf.nn.seq2seq.embedding_rnn_seq2seq(tf.unstack(x), tf.unstack(y), cell, self.config.num_encoder_symbols,
        #                 self.config.num_decoder_symbols, 300)
        outputs, state = tf.nn.seq2seq.basic_rnn_seq2seq(tf.unstack(x), tf.unstack(y), cell)
        preds = tf.sigmoid(state)
        ### END YOUR CODE

        return preds #state # preds

    def add_loss_op(self, preds):
        """Adds ops to compute the loss function.
        Here, we will use a simple l2 loss.

        Tips:
            - You may find the functions tf.reduce_mean and tf.l2_loss
              useful.

        Args:
            pred: A tensor of shape (batch_size, 1) containing the last
            state of the neural network.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        y = self.labels_placeholder

        ### YOUR CODE HERE (~1-2 lines)
        loss = tf.reduce_mean(tf.nn.l2_loss(preds - y))
        ### END YOUR CODE

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        TODO:
            - Get the gradients for the loss from optimizer using
              optimizer.compute_gradients.
            - if self.clip_gradients is true, clip the global norm of
              the gradients using tf.clip_by_global_norm to self.config.max_grad_norm
            - Compute the resultant global norm of the gradients using
              tf.global_norm and save this global norm in self.grad_norm.
            - Finally, actually create the training operation by calling
              optimizer.apply_gradients.
        See: https://www.tensorflow.org/api_docs/python/train/gradient_clipping
        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        ### YOUR CODE HERE (~6-10 lines)
        lossGradients = optimizer.compute_gradients(loss)
        grads = [g[0] for g in lossGradients]
        varss = [g[1] for g in lossGradients]
        if self.config.clip_gradients:
            grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
            lossGradients = zip(grads, varss)
        self.grad_norm = tf.global_norm(grads)
        train_op = optimizer.apply_gradients(lossGradients)
        # - Remember to clip gradients only if self.config.clip_gradients
        # is True.
        # - Remember to set self.grad_norm

        ### END YOUR CODE

        assert self.grad_norm is not None, "grad_norm was not set properly!"
        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.
        This version also returns the norm of gradients.
        """
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def run_epoch(self, sess, train):
        print 'run epoc'
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, batch in enumerate(minibatches(train, self.config.batch_size)):
            print 'are you even listening'
            print i, batch
            loss, grad_norm = self.train_on_batch(sess, *batch)
            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss)])

        return losses, grad_norms

    def fit(self, sess, train):
        losses, grad_norms = [], []
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm = self.run_epoch(sess, train)
            losses.append(loss)
            grad_norms.append(grad_norm)

        return losses, grad_norms

    def __init__(self, config):
        self.config = config
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.grad_norm = None
        self.build()

def generate_sequence(max_length=20, n_samples=9999):
    """
    Generates a sequence like a [0]*n a
    """
    seqs = []
    for _ in range(int(n_samples/2)):
        seqs.append(([[0.,]] + ([[0.,]] * (max_length-1)), [0.]))
        seqs.append(([[1.,]] + ([[0.,]] * (max_length-1)), [1.]))
    return seqs

def test_generate_sequence():
    max_length = 20
    for seq, y in generate_sequence(20):
        assert len(seq) == max_length
        assert seq[0] == y

def make_dynamics_plot(args, x, h, ht_rnn, ht_gru, params):
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')

    Ur, Wr, br, Uz, Wz, bz, Uo, Wo, bo = params

    plt.clf()
    plt.title("""Cell dynamics when x={}:
Ur={:.2f}, Wr={:.2f}, br={:.2f}
Uz={:.2f}, Wz={:.2f}, bz={:.2f}
Uo={:.2f}, Wo={:.2f}, bo={:.2f}""".format(x, Ur[0,0], Wr[0,0], br[0], Uz[0,0], Wz[0,0], bz[0], Uo[0,0], Wo[0,0], bo[0]))

    plt.plot(h, ht_rnn, label="rnn")
    plt.plot(h, ht_gru, label="gru")
    plt.plot(h, h, color='gray', linestyle='--')
    plt.ylabel("$h_{t}$")
    plt.xlabel("$h_{t-1}$")
    plt.legend()
    output_path = "{}-{}-{}.png".format(args.output_prefix, x, "dynamics")
    plt.savefig(output_path)

def compute_cell_dynamics(args):
    with tf.Graph().as_default():
        # You can change this around, but make sure to reset it to 41 when
        # submitting.
        np.random.seed(41)
        tf.set_random_seed(41)

        with tf.variable_scope("dynamics"):
            x_placeholder = tf.placeholder(tf.float32, shape=(None,1))
            h_placeholder = tf.placeholder(tf.float32, shape=(None,1))

            def mat(x):
                return np.atleast_2d(np.array(x, dtype=np.float32))
            def vec(x):
                return np.atleast_1d(np.array(x, dtype=np.float32))

            with tf.variable_scope("cell"):
                Ur, Wr, Uz, Wz, Uo, Wo = [mat(3*x) for x in np.random.randn(6)]
                br, bz, bo = [vec(x) for x in np.random.randn(3)]
                params = [Ur, Wr, br, Uz, Wz, bz, Uo, Wo, bo]

                tf.get_variable("U_r", initializer=Ur)
                tf.get_variable("W_r", initializer=Wr)
                tf.get_variable("b_r", initializer=br)

                tf.get_variable("U_z", initializer=Uz)
                tf.get_variable("W_z", initializer=Wz)
                tf.get_variable("b_z", initializer=bz)

                tf.get_variable("U_o", initializer=Uo)
                tf.get_variable("W_o", initializer=Wo)
                tf.get_variable("b_o", initializer=bo)

            tf.get_variable_scope().reuse_variables()
            y_gru, h_gru = GRUCell(1,1)(x_placeholder, h_placeholder, scope="cell")
            y_rnn, h_rnn = GRUCell(1,1)(x_placeholder, h_placeholder, scope="cell")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)

                x = mat(np.zeros(1000)).T
                h = mat(np.linspace(-3, 3, 1000)).T
                ht_gru = session.run([h_gru], feed_dict={x_placeholder: x, h_placeholder: h})
                ht_rnn = session.run([h_rnn], feed_dict={x_placeholder: x, h_placeholder: h})
                ht_gru = np.array(ht_gru)[0]
                ht_rnn = np.array(ht_rnn)[0]
                #make_dynamics_plot(args, 0, h, ht_rnn, ht_gru, params)

                x = mat(np.ones(1000)).T
                h = mat(np.linspace(-3, 3, 1000)).T
                ht_gru = session.run([h_gru], feed_dict={x_placeholder: x, h_placeholder: h})
                ht_rnn = session.run([h_rnn], feed_dict={x_placeholder: x, h_placeholder: h})
                ht_gru = np.array(ht_gru)[0]
                ht_rnn = np.array(ht_rnn)[0]
                #make_dynamics_plot(args, 1, h, ht_rnn, ht_gru, params)

def make_prediction_plot(args, losses, grad_norms):
    plt.subplot(2, 1, 1)
    plt.title("{} on sequences of length {} ({} gradient clipping)".format(args.cell, args.max_length, "with" if args.clip_gradients else "without"))
    plt.plot(np.arange(losses.size), losses.flatten(), label="Loss")
    plt.ylabel("Loss")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(grad_norms.size), grad_norms.flatten(), label="Gradients")
    plt.ylabel("Gradients")
    plt.xlabel("Minibatch")
    output_path = "{}-{}clip-{}.png".format(args.output_prefix, "" if args.clip_gradients else "no", args.cell)
    plt.savefig(output_path)

def do_sequence_prediction(args):
    # Set up some parameters.
    config = Config()
    config.cell = args.cell
    config.clip_gradients = args.clip_gradients

    # You can change this around, but make sure to reset it to 41 when
    # submitting.
    np.random.seed(41)
    #data = generate_sequence(args.max_length)
    X = np.load('X')
    Y = np.load('Y')
    data = [X, Y]


    with tf.Graph().as_default():
        # You can change this around, but make sure to reset it to 41 when
        # submitting.
        tf.set_random_seed(59)

        # Initializing RNNs weights to be very large to showcase
        # gradient clipping.


        logger.info("Building model...",)
        start = time.time()
        model = SequencePredictor(config)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            losses, grad_norms = model.fit(session, data)

    # Plotting code.
    losses, grad_norms = np.array(losses), np.array(grad_norms)
    #make_prediction_plot(args, losses, grad_norms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs a sequence model to test latching behavior of memory, e.g. 100000000 -> 1')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('predict', help='Plot prediction behavior of different cells')
    command_parser.add_argument('-c', '--cell', choices=['rnn', 'gru', 'lstm'], default='rnn', help="Type of cell to use")
    command_parser.add_argument('-g', '--clip_gradients', action='store_true', default=False, help="If true, clip gradients")
    command_parser.add_argument('-l', '--max-length', type=int, default=20, help="Length of sequences to generate")
    command_parser.add_argument('-o', '--output-prefix', type=str, default="q3", help="Length of sequences to generate")
    command_parser.set_defaults(func=do_sequence_prediction)

    # Easter egg! Run this function to plot how an RNN or GRU map an
    # input state to an output state.
    command_parser = subparsers.add_parser('dynamics', help="Plot cell's dynamics")
    command_parser.add_argument('-o', '--output-prefix', type=str, default="q3", help="Length of sequences to generate")
    command_parser.set_defaults(func=compute_cell_dynamics)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
