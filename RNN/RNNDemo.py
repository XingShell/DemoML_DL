import tensorflow as tf
import numpy as np
from util import data_loads
from util import evaluate

class RNNDemo(object):
    def __init__(self,batch):
        # self.training_iter = 100
        self.is_train = False
        self.lr = 0.01
        self.training_iter = 100000000000
        self.batch_size = batch
        self.n_saves_to_keep = 10
        self.n_inputs = 10
        self.n_steps = 3
        self.n_hidden_unis = 128
        self.n_classes = 2
        self.build_network()

    def build_network(self):
        self.construct_session()
        self.declare_placehoders()
        self.declare_parameters()
        self.compute_cost()
        self.build_optimizer()
        self.initialize_vars()
        self.init_saver()
        self.restore()

    def initialize_vars(self):
        tf.global_variables_initializer().run(session=self.sess)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)

    def restore(self):
        model_file = tf.train.latest_checkpoint('modelsRNN/')
        self.saver.restore(self.sess, model_file)

    def save(self, epoch):
        self.saver.save(self.sess, './modelsRNN/model.ckpt', global_step=epoch)

    def construct_session(self):
        self.sess = tf.Session()

    def declare_placehoders(self):
        with tf.variable_scope('input') as scope:
            self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
            self.y = tf.placeholder(tf.float32, [None, self.n_classes])

    def declare_parameters(self):
        with tf.variable_scope('params') as scope:
            self.weights = {
                'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_unis])),
                'out': tf.Variable(tf.random_normal([self.n_hidden_unis, self.n_classes]))
            }
            self.biases = {
                'in': tf.Variable(tf.constant(0, 1, shape=[self.n_hidden_unis, ])),
                'out': tf.Variable(tf.constant(0, 1, shape=[self.n_classes, ]))
            }
    def prediction(self):
        prediction = self.RNN(self.x, self.weights, self.biases)
        return prediction
    def predict(self, x):
        y = np.ones(2 * len(x)).reshape([-1, 2])
        x = x.reshape(([-1,self.n_steps,self.n_inputs]))
        dtest_test = {self.x: x, self.y: y}
        y = self.sess.run(self.prediction, feed_dict=dtest_test)
        label = [np.argmax(one_hot) for one_hot in y]
        return label


    def RNN(self,X, weights, biases):
        X = tf.reshape(X, [-1, self.n_inputs])
        X_in = tf.matmul(X, weights['in']) + biases['in']
        X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_unis])
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
        _init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
        results = tf.matmul(states[1], weights['out']) + biases['out']
        return results

    def compute_cost(self):
        self.prediction = self.prediction()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.prediction))

    def build_optimizer(self):
        self.train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(self.cost)

    def train_simply(self, batchnum):
        loads = data_loads.Data_Load()
        batch_xs, batch_ys = loads.chooceall()
        x_train, y_train = batch_xs, batch_ys
        x_train = x_train.reshape([-1,self.n_steps,self.n_inputs])
        y_train = y_train.astype(np.int32)
        # y_train, x_train = self.balanceBilabelsData(y_train, x_train)
        labeltrain = tf.expand_dims(y_train, 1)
        indices = tf.expand_dims(tf.range(0, y_train.shape[0], 1), 1)
        concated = tf.concat([indices, labeltrain], 1)
        onehot_labels_train = tf.sparse_to_dense(concated, tf.stack([y_train.shape[0], 2]), 1.0, 0.0)
        dtrain = {self.x: x_train, self.y: self.sess.run(onehot_labels_train)}

        x_test, y_test = loads.all()
        x_test = x_test.reshape([-1, self.n_steps, self.n_inputs])
        y_test = y_test.astype(np.int32)
        labeltest = tf.expand_dims(y_test, 1)
        indicestest = tf.expand_dims(tf.range(0, y_test.shape[0], 1), 1)
        concatedtest = tf.concat([indicestest, labeltest], 1)
        onehot_labels_test = tf.sparse_to_dense(concatedtest, tf.stack([y_test.shape[0], 2]), 1.0, 0.0)
        dtest = {self.x: x_test, self.y: self.sess.run(onehot_labels_test)}

        for i in range(batchnum):
            self.sess.run(self.train_op, feed_dict=dtrain)
            if i % 100 == 0:
                print(self.sess.run(self.cost, feed_dict=dtrain))
            if i % 500 == 0:
                print("------train-------")
                y = self.sess.run(self.prediction, feed_dict=dtrain)
                label = [np.argmax(one_hot) for one_hot in y]
                y_true = self.sess.run(onehot_labels_train)
                label_true = [np.argmax(one_hot) for one_hot in y_true]
                label_true = np.array(label_true)
                label = np.array(label)
                self.save(i // 500)
                label_true.astype(np.int32)
                label.astype(np.int32)
                evaluate.evaluate(label_true, label, simply=True)



if __name__ == '__main__':

    loads = data_loads.Data_Load()
    # batch_xs, batch_ys = loads.chooceall()
    # rnn = RNNDemo(len(batch_ys))
    # rnn.train_simply(2000)
    batch_xs, batch_ys = loads.all()
    rnn2 = RNNDemo(len(batch_ys))
    predict_y = rnn2.predict(batch_xs)
    predict_y = np.array(predict_y)
    evaluate.evaluate(batch_ys, predict_y)

