import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../')
from util import data_loads
from util import evaluate

class DNNDemo(object):


    def add_layer(self,inputs, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        biases = tf.Variable(tf.zeros([1,out_size])+0.1)
        Wx_plus_b = tf.matmul(inputs, Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def __init__(self):
        # self.training_iter = 100
        self.is_train = False
        self.build_network()

#
    def balanceBilabelsData(self,labels,data):
        # num of class 1 << 0
        onenum = 0
        allnum = 0
        listindex = []
        for index, e in enumerate(labels):
            if e == 1:
                onenum += 1
                listindex.append(index)
            allnum += 1
        print(onenum/allnum)
        multyply = ((allnum-onenum)//onenum)-1
        onelabel = np.ones(multyply * onenum, np.int32)
        labels = np.append(onelabel, labels)
        tem = np.tile(data[listindex,], (multyply, 1))
        data = np.concatenate((data, tem), axis=0)
        permutation = np.random.permutation(labels.shape[0])
        data = data[permutation, :]
        labels = labels[permutation]
        return labels, data

    def build_network(self):
        self.construct_session()
        self.declare_placehoders()
        self.declare_parameters()
        self.compute_cost()
        self.build_optimizer()
        self.initialize_vars()
        self.init_saver()
        self.restore()
    def construct_session(self):
        self.sess = tf.Session()
    def declare_placehoders(self):
        self.xs = tf.placeholder(tf.float32, [None, 30])
        self.ys = tf.placeholder(tf.float32, [None, 2])
    def declare_parameters(self):
        with tf.variable_scope('params') as scope:
            self.l1 = self.add_layer(self.xs, 30, 25, activation_function=tf.nn.sigmoid)
            self.l2 = self.add_layer(self.l1, 25, 20, activation_function=tf.nn.sigmoid)
            self.l3 = self.add_layer(self.l2, 20, 15, activation_function=tf.nn.sigmoid)
            self.l4 = self.add_layer(self.l3, 15, 10, activation_function=tf.nn.sigmoid)
            self.l5 = self.add_layer(self.l4, 10, 5, activation_function=tf.nn.sigmoid)
            # l6 = add_layer(l5,60,30,activation_function=tf.nn.sigmoid)
            # l7 = add_layer(l6,30,20,activation_function=tf.nn.sigmoid)
            self.prediction = self.add_layer(self.l5, 5, 2, activation_function=tf.nn.softmax)
    def compute_cost(self):
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.prediction),reduction_indices=[1]))
    def build_optimizer(self):
        # self.global_step = tf.get_variable('global_step',shape=[],initializer=tf.zeros_initializer,trainable=False)
        # if lr_decay_type == 'no_decay':
        #     self.learning_rate = tf.constant()
        self.learning_rate = 0.001
        # optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
    def initialize_vars(self):
        tf.global_variables_initializer().run(session=self.sess)
    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=5)
    def restore(self):
        model_file = tf.train.latest_checkpoint('modelsDNN/')
        self.saver.restore(self.sess, model_file)
    def save(self,epoch):
        self.saver.save(self.sess, './modelsDNN/model.ckpt', global_step=epoch)

    def train_epoch(self,batchnum):
        loads = data_loads.Data_Load()
        batch_xs, batch_ys = loads.all()
        x_train, _, y_train, _ = train_test_split(batch_xs,batch_ys,test_size=0.25)
        x_train, y_train = batch_xs, batch_ys
        y_train = y_train.astype(np.int32)
        y_train, x_train = self.balanceBilabelsData(y_train, x_train)
        labeltrain = tf.expand_dims(y_train, 1)
        indices = tf.expand_dims(tf.range(0, y_train.shape[0], 1), 1)
        concated = tf.concat([indices, labeltrain], 1)
        onehot_labels_train = tf.sparse_to_dense(concated, tf.stack([y_train.shape[0], 2]), 1.0, 0.0)
        dtrain = {self.xs: x_train, self.ys: self.sess.run(onehot_labels_train)}
        x_test, y_test = batch_xs, batch_ys
        y_test = y_test.astype(np.int32)
        labeltest = tf.expand_dims(y_test, 1)
        indicestest = tf.expand_dims(tf.range(0, y_test.shape[0], 1), 1)
        concatedtest = tf.concat([indicestest, labeltest], 1)
        onehot_labels_test = tf.sparse_to_dense(concatedtest, tf.stack([y_test.shape[0], 2]), 1.0, 0.0)
        dtest = {self.xs: x_test, self.ys: self.sess.run(onehot_labels_test)}
        for i in range(batchnum):
            self.sess.run(self.train_step,feed_dict=dtrain)
            if i % 1000 == 0:
                print(self.sess.run(self.cross_entropy,feed_dict=dtrain))
            if i % 5000 == 0:
                print("------train-------")
                y = self.sess.run(self.prediction,feed_dict=dtrain)
                label = [np.argmax(one_hot)for one_hot in y]
                y_true = self.sess.run(onehot_labels_train)
                label_true = [np.argmax(one_hot)for one_hot in y_true]
                label_true = np.array(label_true)
                label = np.array(label)
                self.save(i//50)
                label_true.astype(np.int32)
                label.astype(np.int32)
                evaluate.evaluate(label_true,label,simply=True)
                print('---------test---------')
                y = self.sess.run(self.prediction, feed_dict=dtest)
                label = [np.argmax(one_hot) for one_hot in y]
                y_true = self.sess.run(onehot_labels_test)
                label_true = [np.argmax(one_hot) for one_hot in y_true]
                label_true = np.array(label_true)
                label = np.array(label)
                label_true.astype(np.int32)
                label.astype(np.int32)
                evaluate.evaluate(label_true, label, simply=True)

    def predict(self,x):
        y = np.ones(2*len(x)).reshape([-1,2])
        dtest_test = {self.xs: x, self.ys: y}
        y = self.sess.run(self.prediction, feed_dict=dtest_test)
        label = [np.argmax(one_hot) for one_hot in y]
        return label

    def train_simply(self,batchnum):
        loads = data_loads.Data_Load()

        batch_xs, batch_ys = loads.chooceall()
        x_train, y_train = batch_xs, batch_ys
        y_train = y_train.astype(np.int32)
        # y_train, x_train = self.balanceBilabelsData(y_train, x_train)
        labeltrain = tf.expand_dims(y_train, 1)
        indices = tf.expand_dims(tf.range(0, y_train.shape[0], 1), 1)
        concated = tf.concat([indices, labeltrain], 1)
        onehot_labels_train = tf.sparse_to_dense(concated, tf.stack([y_train.shape[0], 2]), 1.0, 0.0)
        dtrain = {self.xs: x_train, self.ys: self.sess.run(onehot_labels_train)}

        x_test, y_test = loads.all()
        y_test = y_test.astype(np.int32)
        labeltest = tf.expand_dims(y_test, 1)
        indicestest = tf.expand_dims(tf.range(0, y_test.shape[0], 1), 1)
        concatedtest = tf.concat([indicestest, labeltest], 1)
        onehot_labels_test = tf.sparse_to_dense(concatedtest, tf.stack([y_test.shape[0], 2]), 1.0, 0.0)
        dtest = {self.xs: x_test, self.ys: self.sess.run(onehot_labels_test)}

        for i in range(batchnum):
            self.sess.run(self.train_step,feed_dict=dtrain)
            if i % 1000 == 0:
                print(self.sess.run(self.cross_entropy,feed_dict=dtrain))
            if i % 5000 == 0:
                print("------train-------")
                y = self.sess.run(self.prediction,feed_dict=dtrain)
                label = [np.argmax(one_hot)for one_hot in y]
                y_true = self.sess.run(onehot_labels_train)
                label_true = [np.argmax(one_hot)for one_hot in y_true]
                label_true = np.array(label_true)
                label = np.array(label)
                self.save(i//50)
                label_true.astype(np.int32)
                label.astype(np.int32)
                evaluate.evaluate(label_true,label,simply=True)
                print('---------test---------')
                y = self.sess.run(self.prediction, feed_dict=dtest)
                label = [np.argmax(one_hot) for one_hot in y]
                y_true = self.sess.run(onehot_labels_test)
                label_true = [np.argmax(one_hot) for one_hot in y_true]
                label_true = np.array(label_true)
                label = np.array(label)
                label_true.astype(np.int32)
                label.astype(np.int32)
                evaluate.evaluate(label_true, label, simply=True)





if __name__=='__main__':
    dnn = DNNDemo()
    dnn.train_simply(10000000)
    loads = data_loads.Data_Load()
    # batch_xs, batch_ys = loads.all()
    batch_xs, batch_ys = loads.chooceall()
    predict_y = dnn.predict(batch_xs)
    predict_y = np.array(predict_y)
    evaluate.evaluate(batch_ys, predict_y)
    # predict_y = DNNDemo().predict(batch_xs)
    # evaluate.evaluate(batch_ys,predict_y)
    # print(predict_y)



