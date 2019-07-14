import tensorflow as tf
import data_loads
import os
import numpy as np
import evaluate

lr = 0.01
training_iter = 100000000000
batch_size = 5000
batch_size = 21969
n_saves_to_keep = 10
n_inputs = 10
n_steps = 5
n_hidden_unis = 128
n_classes = 2

x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
           }
biases = {
    'in':tf.Variable(tf.constant(0,1,shape=[n_hidden_unis,])),
    'out':tf.Variable(tf.constant(0,1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    X = tf.reshape(X, [-1,n_inputs])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps,n_hidden_unis])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state =_init_state, time_major = False)

    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results


pred = RNN(x,weights,biases)
correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y, 1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdadeltaOptimizer(lr).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == '__main__':
    with tf.Session() as sess:
        loads = data_loads.Data_Load()
        step = 0
        is_train = False
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=n_saves_to_keep)
        if is_train:
            # model_file = tf.train.latest_checkpoint('./Models/')
            # saver.restore(sess, model_file)
            while step*batch_size < training_iter:
                batch_xs, batch_ys = loads.next_batch(batch_size)
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
                batch_ys = batch_ys.astype(np.int32)
                label1 = tf.expand_dims(batch_ys,1)
                indices = tf.expand_dims(tf.range(0,batch_size,1),1)
                concated = tf.concat([indices,label1],1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size,n_classes]),1.0,0.0)
                d = feed_dict={
                    x:batch_xs,
                    y:sess.run(onehot_labels)
                }
                sess.run([train_op], feed_dict=d)
                if step % 100 == 0:
                    _y_labels = sess.run(tf.arg_max(pred, 1), feed_dict=d)
                    print(sess.run(accuracy, feed_dict=d), end="----")
                    print(sess.run(cost, feed_dict=d))
                    saver.save(sess, './Models/rnnmodel.ckpt',global_step=step//100)
                step += 1
        else:
            batch_xs, batch_ys = loads.all()
            batch_size = len(batch_ys)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            batch_ys = batch_ys.astype(np.int32)
            label1 = tf.expand_dims(batch_ys, 1)
            indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
            concated = tf.concat([indices, label1], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([len(batch_ys), n_classes]), 1.0, 0.0)
            d = feed_dict = {
                x: batch_xs,
                y: sess.run(onehot_labels)
            }
            model_file = tf.train.latest_checkpoint('./Models/')
            saver.restore(sess, model_file)
            _y_labels = sess.run(tf.arg_max(pred, 1), feed_dict=d)
            print(sess.run(accuracy, feed_dict=d), end="----")
            print(sess.run(cost, feed_dict=d))
            evaluate.evaluate(batch_ys,_y_labels)
