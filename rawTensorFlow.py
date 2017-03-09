import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, max_output_length,max_input_length,embed_size, embeddings, hidden_size, n_classes, batch_size, lr):
        self.hidden_size = hidden_size
        self.max_output_length = max_output_length
        self.max_input_length = max_input_length
        self.embed_size = embed_size
        self.pretrained_embeddings = embeddings
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.lr = lr
    def create_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.int32, (None, self.max_input_length))
        self.labels_placeholder = tf.placeholder(tf.int32, (None, self.max_output_length))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_output_length))
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self,inputs_batch, mask_batch, dropout, labels):
        feed_dict = {
            self.input_placeholder : inputs_batch,
            self.mask_placeholder : mask_batch,
            self.dropout_placeholder: dropout,
            self.labels_placeholder: labels
        }
        return feed_dict

    def add_embedding(self):
        lookup = tf.nn.embedding_lookup(tf.Variable(self.pretrained_embeddings), self.input_placeholder)
        embeddings = tf.reshape(lookup, (-1, self.max_input_length, self.embed_size))
        ### END YOUR CODE
        return embeddings

    def output_embedding(self):
        lookup = tf.nn.embedding_lookup(tf.Variable(self.pretrained_embeddings), self.labels_placeholder)
        embeddings = tf.reshape(lookup, (-1, self.max_output_length, self.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        y = self.output_embedding()
        preds = [] # Predicted output at each timestep should go here!


        EncoderCell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        DecoderCell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        U = tf.get_variable('U', (self.hidden_size, self.n_classes), tf.float32, tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable('b_2', (self.n_classes,), tf.float32, tf.constant_initializer(0.0))
        h_c = tf.zeros((self.batch_size,self.hidden_size), tf.float32)
        h_m = tf.zeros((self.batch_size,self.hidden_size), tf.float32)
        h = (h_c, h_m)
        for time_step in range(self.max_input_length):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            o_t, h = EncoderCell(x[:,time_step,:], h)

        for time_step in range(self.max_output_length):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            o_t, h = EncoderCell(y[:,time_step,:], h)
            o_drop_t = tf.nn.dropout(o_t, dropout_rate)
            y_t = tf.matmul(o_drop_t, U)+ b_2
            preds.append(y_t)
        # Make sure to reshape @preds here.
        preds = tf.pack(preds,1)
        ### YOUR CODE HERE (~2-4 lines)
        ### END YOUR CODE

        #assert preds.get_shape().as_list() == [None, self.max_output_length, self.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_output_length, self.n_classes], preds.get_shape().as_list())
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        sftmx_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
        loss = tf.reduce_mean(tf.boolean_mask(sftmx_ce,self.mask_placeholder))
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

def main():
    X = np.load('X')
    Y = np.load('Y')
    mask = np.load('outputMask')
    embeddings = np.load('embeddings').astype(np.float32)
    model = Model(31,31,300, embeddings, 128, 30033, 10, 0.001)
    model.create_placeholders()
    preds = model.add_prediction_op()
    loss = model.add_loss_op(preds)
    training_op = model.add_training_op(loss)
    init = tf.global_variables_initializer()
    batch_size = 10
    count = 0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(int(X.shape[0]/batch_size)):
            feed = model.create_feed_dict(X[batch_size*i:batch_size*(i+1), :],mask[batch_size*i:batch_size*(i+1), :],1,Y[batch_size*i:batch_size*(i+1), :])
            newLoss, train = sess.run([loss,training_op], feed_dict = feed)
            print newLoss
            print count
            count += 1
main()
