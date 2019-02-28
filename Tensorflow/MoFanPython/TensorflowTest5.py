__doc__ = 'description'
__author__ = '13314409603@163.com'


import tensorflow as tf

#创建op，使用的是默认图
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2]])
product = tf.matmul(matrix1,matrix2)

#创建会话，启动默认图
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
if __name__ == '__main__':
    pass