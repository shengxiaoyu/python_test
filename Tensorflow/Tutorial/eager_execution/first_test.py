#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
import numpy as np
import sys
import tempfile

def main():
    tf.enable_eager_execution()
    # print(tf.encode_base64('你好'))
    # print(np.ones([3,3]))
    x = tf.random_uniform([5])
    print(x)
    x = tf.random_uniform([3, 3])

    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())

    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))
    tensors = tf.ones([5],dtype=tf.int32)
    t1 = tf.data.Dataset.from_tensors(tensors)
    for x in t1:
        print(np.add(x, 1))
    ds_tensors = tf.data.Dataset.from_tensor_slices(tensors)
    for x in ds_tensors:
        print(np.add(x,1))

    # Create a CSV file

    _, filename = tempfile.mkstemp()

    with open(filename, 'w') as f:
      f.write("""Line 1
    Line 2
    Line 3
      """)

    # ds_file = tf.data.TextLineDataset(filename)
    path = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\NERdata\\train\\(2017)津0103民初12551号.txt'
    dataset = tf.data.TextLineDataset(path)
    for x in dataset:
        print(x)
if __name__ == '__main__':
    main()
    sys.exit(0)