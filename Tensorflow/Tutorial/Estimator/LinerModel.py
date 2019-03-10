#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools

__doc__ = 'description'
__author__ = '13314409603@163.com'
import tensorflow as tf
from tensorflow import feature_column as fc
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output
import argparse
import pandas as pd


FLAGS = None
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]
_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}
def main():

    # train_df = pd.read_csv(FLAGS.train_file,header = None ,names = _CSV_COLUMNS)
    # test_df = pd.read_csv(FLAGS.test_file,header = None ,names = _CSV_COLUMNS)

    # ds = input_fn(FLAGS.train_file,5,True,10)

    train_inpf = functools.partial(input_fn,FLAGS.train_file,num_epochs =2,shuffle=True,batch_size=64)
    test_inpf = functools.partial(input_fn,FLAGS.test_file,num_epochs =1,shuffle=False,batch_size=64)

    age = fc.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')


    relationship = fc.categorical_column_with_vocabulary_list('relationship',['Husband','Not-in-family','Wife','Own-child','Unmarried','Other-relative'])

    occupation = fc.categorical_column_with_hash_bucket('occupation',hash_bucket_size=1000)
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


    # my_numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]
    # my_categorical_columns = [relationship, occupation, education, marital_status, workclass]
    # classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns+my_categorical_columns)

    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]
    classifier = tf.estimator.LinearClassifier(
        feature_columns=base_columns + crossed_columns,
        optimizer=tf.train.FtrlOptimizer(learning_rate=0.1))
    classifier.train(train_inpf)
    result = classifier.evaluate(test_inpf)

    clear_output()
    print(result)

#简易版input_fn，这种方式需要把数据一次性读入内存，较大的数据集应从磁盘流式传输
def easy_input_function(df,label_key,num_epochs,shuffle,batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

    if shuffle :
        ds = ds.shuffle(10000)

    ds = ds.repeat(num_epochs).batch(batch_size)

    return ds


def input_fn(data_file,num_epochs,shuffle,batch_size):
    def parse_csv(value):
        columns = tf.decode_csv(value,_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS,columns))
        labels = features.pop('income_bracket')
        classes = tf.equal(labels,'>50K')
        return features,classes

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv,num_parallel_calls=5)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    tf.data.TextLineDataset
    return dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="美国人口普查预测")
    parser.add_argument('--train_file',help='train file',default='E:\\pyWorkspace\\test\\Tensorflow\\Tutorial\\Estimator\\adult.data')
    parser.add_argument('--test_file',help='test file',default='E:\\pyWorkspace\\test\\Tensorflow\\Tutorial\\Estimator\\adult.test')
    FLAGS,_ = parser.parse_known_args()
    main()
    sys.exit(0)