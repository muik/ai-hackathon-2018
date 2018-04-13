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
import time
from multiprocessing import Pool

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

from kor_char_parser import decompose_str_as_one_hot

def group_count(name, items):
    df = pd.DataFrame(data={name: items})
    df = df.groupby([name]).size().reset_index(name='counts')
    total = len(items)
    df['percent'] = df['counts'].apply(lambda x: round(x * 100 / total, 1))
    return df

class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int, max_size=-1):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        print('pandas version:', pd.__version__)
        if max_size > -1:
            print('max dataset size:', max_size)

        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')

        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.reviews, self.lengths = preprocess(f.readlines()[:max_size], max_length)

        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = [np.float32(x.rstrip()) for x in f.readlines()[:max_size]]

        # 라벨별 비중
        #print(group_count('label', self.labels))

    def get_sampler(self):
        test_size = 0.2
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        X = np.arange(len(self))
        try:
            train_index, eval_index = next(sss.split(X, self.labels))
        except ValueError as e:
            if not 'The least populated class in y has only ' in str(e):
                raise e
            print('Use just ShuffleSplit')
            from sklearn.model_selection import ShuffleSplit
            sss = ShuffleSplit(n_splits=1, test_size=test_size)
            train_index, eval_index = next(sss.split(X, self.labels))

        train_sampler = SubsetRandomSampler(train_index)
        eval_sampler = SubsetRandomSampler(eval_index)
        return train_sampler, eval_sampler

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.reviews[idx], self.lengths[idx], self.labels[idx]

def preprocess(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    t0 = time.time()
    with Pool(12) as p:
        vectorized_data = p.map(decompose_str_as_one_hot, [datum.strip() for datum in data])
    print("vectorized_data loaded %.2f s" % (time.time() - t0))

    # one hot length
    #df = pd.DataFrame(data={'vectorized_data_length': vec_data_lengths})
    #print(df.describe(percentiles=[0.95, 0.997]))

    t0 = time.time()
    total_count = len(data)
    zero_padding = np.zeros((total_count, max_length), dtype=np.int32)
    vec_data_lengths = np.zeros((total_count), dtype=np.int32)

    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        seq = [x+1 for x in seq]
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
        vec_data_lengths[idx] = length
    print("zero_padding loaded %.2f s" % (time.time() - t0))
    return zero_padding, vec_data_lengths
