__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:7,input2:2}))


if __name__ == '__main__':
    pass