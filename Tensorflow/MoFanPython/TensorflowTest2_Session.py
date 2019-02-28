__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
matrix3 = tf.constant([[0,0],[0,20]])
product = tf.matmul(matrix1,matrix2)

#method1
sess =  tf.Session()
result = sess.run(product)
print(result)
sess.close()

#method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

    print(sess.run(matrix3))

if __name__ == '__main__':
    pass