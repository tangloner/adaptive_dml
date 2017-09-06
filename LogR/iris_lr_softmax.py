#
#   iris_lr_softmax.py
#       date. 5/6/2016
#       IRIS data set classification
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def prep_data(train_siz=120, test_siz=30):
    '''
      class: 
        1. Iris Setosa, 2. Iris Versicolor, 3. Iris Virginica
    '''
    cols = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']  
    iris_df = pd.read_csv('iris.data', header=None, names=cols)
    
    # Encode class 
    class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    iris_df['iclass'] = [class_name.index(class_str) 
                            for class_str in iris_df['class'].values]   
    # Random Shuffle before split to train/test
    data_len = len(iris_df)
    orig = np.arange(data_len)
    perm = np.copy(orig)
    np.random.shuffle(perm)
    iris = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'iclass']].values
    iris[orig, :] = iris[perm, :]
    
    # generate onehot label data
    label = np.zeros((data_len, 3), dtype=np.float32)
    for i in range(data_len):
      iclass = int(iris[i, -1])
      label[iclass] = 1.0
    
    # Split dataset
    trX = iris[:train_siz, :-1]
    teX = iris[train_siz:, :-1]
    trY = label[:train_siz, :]
    teY = label[train_siz:, :]
    
    return trX, trY, teX, teY
    
def linear_model(X, w, b):
    output = tf.matmul(X, w) + b

    return output

if __name__ == '__main__':
    tr_x, tr_y, te_x, te_y = prep_data()

    # Variables
    x = tf.placeholder(tf.float32, [None, 4])
    y_ = tf.placeholder(tf.float32, [None, 3])
    
    w = tf.Variable(tf.random_normal([4, 3], mean=0.0, stddev=0.05))
    b = tf.Variable(tf.zeros([3]))

    y_pred = linear_model(x, w, b)
    y_pred_softmax = tf.nn.softmax(y_pred)   # for prediction

    loss = -tf.reduce_sum(y_*tf.log(y_pred_softmax))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred_softmax, 1), 
        tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('Training...')
        i = 0
        while True:
#        for i in range(10001):
            batch_xs, batch_ys = tr_x, tr_y
            train_step.run({x: batch_xs, y_: batch_ys})                 
        
            if i % 1000 == 0:
                fd_train = {x: batch_xs, y_: batch_ys}
                loss_step = loss.eval(fd_train)
                train_accuracy = accuracy.eval(fd_train)
                print('  step, loss, accurary = %6d: %8.3f,%8.3f' % (i, loss_step, train_accuracy))
            # Test trained model
            fd_test = {x: te_x, y_: te_y}
            print('accuracy = %10.4f' % accuracy.eval(fd_test))
            if accuracy.eval(fd_test) >= 0.9:
                break


