# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Bidirectional, LSTM

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess
from util import local_save, local_load

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        data1, data2 = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output_sigmoid, feed_dict={x1: data1, x2: data2})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=100)
    args.add_argument('--strmaxlen', type=int, default=84)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    hidden_layer_size = 20
    learning_rate = 0.001
    character_size = 251

    x1 = tf.placeholder(tf.int32, [None, config.strmaxlen])
    x2 = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])

    def sentence_embedding(x):
        embedded = tf.nn.embedding_lookup(char_embedding, x)
        with tf.variable_scope('sentence', reuse=tf.AUTO_REUSE):
            embedded = Bidirectional(LSTM(config.embedding))(embedded)
            return embedded
            hidden_layer = tf.reshape(embedded, [-1, input_size])
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, 200)
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, 100)
            hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, 20)
        return hidden_layer

    sentence1 = sentence_embedding(x1)
    sentence2 = sentence_embedding(x2)

    sentence1 = tf.nn.l2_normalize(sentence1, 1)
    sentence2 = tf.nn.l2_normalize(sentence2, 1)

    # cosine similarity
    dot = tf.reduce_sum(tf.multiply(sentence1, sentence2), 1)
    norm1 = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(sentence1), 1)), 0.000000001)
    norm2 = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(sentence2), 1)), 0.000000001)
    similarity = dot / (norm1 * norm2)

    output_sigmoid = tf.sigmoid(similarity - 0.8)

    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(output_sigmoid)) - (1-y_) * tf.log(1-output_sigmoid))

    # l2 loss
    t_vars = [v for v in tf.trainable_variables() if not 'char_embedding' in v.name]
    l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in t_vars ])
    loss_op = binary_cross_entropy + l2_loss * 0.0001

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    prediction = tf.cast(output_sigmoid > config.threshold, dtype=tf.float32)
    accuracy_op, accuracy_updates = tf.metrics.accuracy(y_, prediction)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            next_element = dataset.get_next(config.batch)
            for i in range(one_batch_size):
                data1, data2, labels = sess.run(next_element)
                _, loss = sess.run([train_step, loss_op],
                        feed_dict={x1: data1, x2: data2, y_: labels})
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)

            # Accuracy
            accuracies = []
            next_element = dataset.get_next(config.batch)
            for i in range(one_batch_size // 10):
                data1, data2, labels = sess.run(next_element)
                _, accuracy = sess.run([accuracy_updates, accuracy_op],
                        feed_dict={x1: data1, x2: data2, y_: labels})
                accuracies.append(accuracy)
            accuracy = np.mean(np.array(accuracies))

            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size),
                'accuracy:', accuracy)
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

            if not HAS_DATASET and not IS_ON_NSML:
                local_save(sess, epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        if not HAS_DATASET and not IS_ON_NSML:
            local_load(sess)
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(temp_res)