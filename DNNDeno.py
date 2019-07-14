import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import data_loads
import evaluate



def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


xs = tf.placeholder(tf.float32,[None,50])
ys = tf.placeholder(tf.float32,[None,2])

l1 = add_layer(xs,50,100,activation_function=tf.nn.sigmoid)
l2 = add_layer(l1,100,150,activation_function=tf.nn.sigmoid)
l3 = add_layer(l2,150,100,activation_function=tf.nn.sigmoid)
l4 = add_layer(l3,100,80,activation_function=tf.nn.sigmoid)
l5 = add_layer(l4,80,60,activation_function=tf.nn.sigmoid)
l6 = add_layer(l5,60,30,activation_function=tf.nn.sigmoid)
l7 = add_layer(l6,30,20,activation_function=tf.nn.sigmoid)
prediction = add_layer(l7,20,2,activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

#
# def balanceBilabelsData(labels,data):
#     # num of class 1 << 0
#     onenum = 0
#     allnum = 0
#     listindex = []
#     for index, e in enumerate(labels):
#         if e == 1:
#             onenum += 1
#             listindex.append(index)
#         allnum += 1
#     print(onenum/allnum)
#     multyply = ((allnum-onenum)//onenum)-1
#     onelabel = np.ones(multyply * onenum, np.int32)
#     labels = np.append(onelabel, labels)
#     tem = np.tile(data[listindex,], (multyply, 1))
#     data = np.concatenate((data, tem), axis=0)
#     permutation = np.random.permutation(labels.shape[0])
#     data = data[permutation, :]
#     labels = labels[permutation]
#     return labels, data
# # 数据
loads = data_loads.Data_Load()
batch_xs, batch_ys = loads.all()
x_train, x_test, y_train, y_test = train_test_split(batch_xs,batch_ys,test_size=0.25)
y_test = y_test.astype(np.int32)
y_train = y_train.astype(np.int32)
# y_train,x_train = balanceBilabelsData(y_train,x_train)
listindex = []
labeltrain = tf.expand_dims(y_train,1)
indices = tf.expand_dims(tf.range(0,y_train.shape[0],1),1)
concated = tf.concat([indices,labeltrain],1)
onehot_labels_train = tf.sparse_to_dense(concated, tf.stack([y_train.shape[0],2]),1.0,0.0)
#
# y_test,x_test = balanceBilabelsData(y_test,x_test)
labeltest = tf.expand_dims(y_test,1)
indices = tf.expand_dims(tf.range(0,y_test.shape[0],1),1)
concated = tf.concat([indices,labeltest],1)
onehot_labels_test = tf.sparse_to_dense(concated, tf.stack([y_test.shape[0],2]),1.0,0.0)




training_iter = 100000
is_train = True
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=5)
    dtrain = {xs: x_train, ys: sess.run(onehot_labels_train)}
    dtest={xs:x_test, ys:sess.run(onehot_labels_test)}
    if is_train:
        # model_file = tf.train.latest_checkpoint('./Models/modelsDNN/')
        # saver.restore(sess, model_file)
        for i in range(training_iter):
            sess.run(train_step,feed_dict=dtrain)
            if i % 100==0:
                print(sess.run(cross_entropy,feed_dict=dtrain))
            if i % 1000 == 0:
                train_y_labels = sess.run(tf.arg_max(prediction, 1), feed_dict=dtrain)
                test_y_labels = sess.run(tf.arg_max(prediction, 1), feed_dict=dtest)
                print("------train-------")
                evaluate.evaluate(y_train,train_y_labels)
                print("------test-------")
                evaluate.evaluate(y_test,test_y_labels)
                saver.save(sess, './modelsDNN/model.ckpt', global_step=i//1000)
    else:
        model_file = tf.train.latest_checkpoint('./Models/modelsDNN/')
        saver.restore(sess, model_file)
        batch_xs, batch_ys = loads.all()
        print(sess.run(cross_entropy,feed_dict=dtest))
        predictionLabel =  sess.run(tf.arg_max(prediction, 1), feed_dict=dtrain)
        evaluate.evaluate(batch_ys,predictionLabel)