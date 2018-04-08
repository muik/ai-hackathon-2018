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
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data, data_lengths = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data, data_lengths)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    length = []
    label = []
    for datum in data:
        review.append(datum[0])
        length.append(datum[1])
        label.append(datum[2])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(length), np.array(label)


class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int, dropout_prob: float,
            rnn_layers: int):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        self.num_layers = rnn_layers
        D = self.embedding_dim
        H = self.embedding_dim
        self.hidden_size = H
        self.rnn_dim = H * 2
        self.lstm = nn.LSTM(D, H, self.num_layers, batch_first=False, bidirectional=True,
                dropout=dropout_prob)

        self.attention_matrix = nn.Sequential(
            nn.Linear(H*2, H),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(H, 1),
        )
        self.attention_vector = nn.Softmax(dim=1)

        # 첫 번째 레이어
        self.fc = nn.Sequential(
            nn.Linear(H*2, H),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(H, 1),
        )

    def forward(self, data: list, lengths: list):
        """

        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())

        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()

        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        mask = (data_in_torch > 0).unsqueeze(-1).float()
        embeds = embeds * mask
        embeds = torch.transpose(embeds, 1, 0)
        output, (hn, _) = self.lstm(embeds)
        last_h = torch.cat((hn[-2], hn[-1]), dim=1)
        output = torch.transpose(output, 1, 0)
        output = output * mask

        # https://github.com/wballard/mailscanner/blob/attention/mailscanner/layers/attention.py
        attn_mat = self.attention_matrix(output)
        attn_vec = self.attention_vector(attn_mat.view(batch_size, -1))
        attn_vec = attn_vec.unsqueeze(2).expand(batch_size, self.max_length, self.rnn_dim)
        context = output * attn_vec
        hidden = torch.sum(context, dim=1)

        # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
        output = torch.sigmoid(self.fc(hidden)) * 9 + 1
        return output


if __name__ == '__main__':
    print('torch version:', torch.__version__)
    print('numpy version:', np.__version__)
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=1000)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding', type=int, default=32)
    args.add_argument('--dropout', type=float, default=0.5)
    args.add_argument('--rnn_layers', type=int, default=2)
    config = args.parse_args()

    if config.mode == 'train':
        dropout_prob = config.dropout
        is_training = True
    else:
        dropout_prob = 0.0
        is_training = False

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    model = Regression(config.embedding, config.strmaxlen, dropout_prob,
            config.rnn_layers)
    if GPU_NUM:
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_accuracy = 0.0
            for i, (data, lengths, labels) in enumerate(train_loader):
                # 아래코드 때문에 학습이 제대로 안된다. 알 수 없음
                #sorted_index = np.argsort(-lengths)
                #data = np.array(data)[sorted_index]
                #lengths = lengths[sorted_index]

                predictions = model(data, lengths)
                label_vars = Variable(torch.from_numpy(labels))
                if GPU_NUM:
                    label_vars = label_vars.cuda()
                loss = criterion(predictions, label_vars)
                if GPU_NUM:
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct = label_vars.eq(torch.round(predictions.view(-1)))
                accuracy = (correct.sum().data[0] / len(labels))

                if i % 100 == 0:
                    print('Batch : ', i + 1, '/', total_batch,
                          ', MSE in this minibatch: ', round(loss.data[0], 3),
                          ', Accuracy:', round(accuracy, 2))
                avg_loss += loss.data[0]
                avg_accuracy += accuracy

            accuracy = round(float(avg_accuracy / total_batch), 2)
            print('epoch:', epoch, ' train_loss:', round(float(avg_loss/total_batch), 3),
                    ' accuracy:', accuracy)
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            #
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/total_batch), step=epoch, accuracy=accuracy)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)