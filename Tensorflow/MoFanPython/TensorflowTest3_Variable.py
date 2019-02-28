__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf

state = tf.Variable(0,name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.initialize_all_variables() #!must initialize variable if have defined variables

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
if __name__ == '__main__':
    pass