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


import os
from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf

from kor_char_parser import decompose_str_as_one_hot


class KinQueryDataset:
    """
        지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int):
        """

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        with open(queries_path, 'rt', encoding='utf8') as f:
            self.queries1, self.queries2, self.q1_lengths, self.q2_lengths = preprocess(f.readlines(), max_length)
        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

        # 같은 질문 데이터셋 추가
        #line_numbers = self.expand_same_set()

        #print('추가된 같은 질문 데이터셋 확인 (상위 50개)')
        #with open(queries_path, 'rt', encoding='utf8') as f:
        #    same_set = np.array(f.readlines())[line_numbers[:50]]
        #    print(same_set)

        self.dataset = tf.data.Dataset.from_tensor_slices((
            self.queries1, self.queries2, self.q1_lengths, self.q2_lengths, self.labels))

        # 데이터 분석
        sentence_lengths = []
        word_counts = []
        with open(queries_path, 'rt', encoding='utf8') as f:
            for line in f:
                #print(line.rstrip())
                s1, s2 = line.strip().split("\t")
                sentence_lengths.append(len(s1))
                sentence_lengths.append(len(s2))
                word_counts.append(len(s1.split()))
                word_counts.append(len(s2.split()))
        df = pd.DataFrame(data={'sentence_length': sentence_lengths, 'word_count': word_counts})
        print(df.describe(percentiles=[0.95, 0.99]))

        #with open(labels_path) as f:
        #    for label in f:
        #        print(label.rstrip())
        labels = self.labels.flatten()
        print('0 labels count:', (labels == 0).sum())
        print('1 labels count:', (labels == 1).sum())

    def expand_same_set(self):
        pre_q = np.array([])
        pre_q_len = 0
        same_queries = []
        extra_data = []
        line_numbers = []

        same_index = np.squeeze(self.labels) == 1.0
        lines = np.arange(self.labels.shape[0])

        for q1, q2, q1_len, q2_len, line_num in zip(self.queries1[same_index], self.queries2[same_index],
                self.q1_lengths[same_index], self.q2_lengths[same_index], lines[same_index]):
            if pre_q_len == q1_len and np.array_equal(pre_q, q1):
                same_queries.append((q2, q2_len, line_num))
                continue
            if len(same_queries) >= 2:
                extra_data.extend(list(combinations(same_queries, 2)))
                line_numbers.extend([n for _,_,n in same_queries])
            pre_q = q1
            pre_q_len = q1_len
            same_queries = [(q2, q2_len, line_num)]

        if len(same_queries) >= 2:
            extra_data.extend(list(combinations(same_queries, 2)))

        print("extra data count:", len(extra_data))
        if not extra_data:
            return

        extra_q1 = np.array([q1[0] for q1, _ in extra_data])
        extra_q2 = np.array([q2[0] for _, q2 in extra_data])
        extra_q1_len = np.array([q1[1] for q1, _ in extra_data])
        extra_q2_len = np.array([q2[1] for _, q2 in extra_data])
        self.queries1 = np.vstack((self.queries1, extra_q1))
        self.queries2 = np.vstack((self.queries2, extra_q2))
        self.labels = np.vstack((self.labels, np.ones([extra_q1.shape[0], 1], dtype=np.float32)))
        self.q1_lengths = np.concatenate((self.q1_lengths, extra_q1_len))
        self.q2_lengths = np.concatenate((self.q2_lengths, extra_q2_len))
        return line_numbers

    def get_next(self, batch_size):
        return self.dataset.shuffle(1000).batch(batch_size) \
                .make_one_shot_iterator().get_next()

    def make_initializable_iterator(self, batch_size):
        return self.dataset.shuffle(1000).repeat().batch(batch_size) \
                .make_initializable_iterator()

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.queries1)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.queries1[idx], self.queries2[idx], self.labels[idx]


def preprocess(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    vectorized_data1 = []
    vectorized_data2 = []
    vectorized_data1_lengths = []
    vectorized_data2_lengths = []

    for datum in data:
        s1, s2 = datum.strip().split("\t")
        onehot1 = decompose_str_as_one_hot(s1, warning=False)
        onehot2 = decompose_str_as_one_hot(s2, warning=False)
        vectorized_data1.append(onehot1)
        vectorized_data2.append(onehot2)
        vectorized_data1_lengths.append(len(onehot1))
        vectorized_data2_lengths.append(len(onehot2))

    # one hot length
    #v_data_lens = [len(x) for x in (vectorized_data1 + vectorized_data2)]
    #df = pd.DataFrame(data={'vectorized_data_length': v_data_lens})
    #print(df.describe(percentiles=[0.95, 0.997]))

    def to_zero_padded(vectorized_data):
        zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
        for idx, seq in enumerate(vectorized_data):
            length = len(seq)
            if length >= max_length:
                length = max_length
                zero_padding[idx, :length] = np.array(seq)[:length]
            else:
                zero_padding[idx, :length] = np.array(seq)
        return zero_padding

    zero_padding1 = to_zero_padded(vectorized_data1)
    zero_padding2 = to_zero_padded(vectorized_data2)

    return zero_padding1, zero_padding2, np.array(vectorized_data1_lengths), np.array(vectorized_data2_lengths)
