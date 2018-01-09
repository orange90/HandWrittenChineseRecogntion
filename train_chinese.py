import tensorflow as tf
import numpy as np
from read_data import ChineseWrittenChars
import time
import sys
chars = ChineseWrittenChars()

def add_layer(inputs, in_size, out_size,n_layer, activation_function = None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weight = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weight', Weight)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weight)+biases
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/output', output)
        return output

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,4096],name = 'x_input') #change according to image size, this is 64*64
    ys = tf.placeholder(tf.float32,[None,100],name = 'y_input') #change according to output size, this is 100

# l1 = add_layer(xs,1,10, n_layer=1, activation_function=tf.nn.softmax)

prediction = add_layer(xs,4096,100,n_layer=2, activation_function=tf.nn.softmax)#change according to image size, this is 64*64

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1])) # loss


# with tf.name_scope('loss'):
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    # tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/',sess.graph)
init = tf.initialize_all_variables()

sess.run(init)

start_load_time = time.time()
print 'start loading test images......'
test_images,test_label = chars.test.load_all(flatten=True)
# here I printed memory usage and loading time,
# you cantune memory and batch size according to you machine memory.
print 'successfully loaded test images......used time %f, test images memory %d'%(time.time()-start_load_time, sys.getsizeof(test_images))

for i in range(9000):
    batch_size = 100
    start_load_time = time.time()
    batch_xs, batch_ys = chars.train.load_next_batch(batch_size,flatten=True)
    print 'successfully loaded batch %d images......used time %f, batch images memory %d'%(batch_size, time.time()-start_load_time, sys.getsizeof(batch_xs))
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    print 'iter %d'%i
    if i%10 == 0:
        print (compute_accuracy(test_images,test_label))