import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int, dropout_prob: float,
            rnn_layers: int, use_gpu: bool, model_type: str):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.character_size = 252
        self.output_dim = 1  # Regression
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim, padding_idx=0)

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

        self.attn = nn.Sequential(
            nn.Linear(max_length*H*2, max_length),
            nn.Softmax(dim=1),
            nn.Dropout(p=dropout_prob),
        )

        self.fc = nn.Sequential(
            nn.Linear(H*4, H),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(H, int(H/2)),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(int(H/2), 1),
        )

        self.model_type = model_type
        print('model type:', self.model_type)

        def conv_layer(in_ch, out_ch, kernel):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel),
                nn.ReLU(),
                nn.MaxPool1d(kernel, stride=kernel),
                )
        def conv_layer2(in_ch, out_ch, kernel):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch*2, kernel),
                nn.ReLU(),
                nn.MaxPool1d(kernel, stride=kernel),
                nn.Conv1d(out_ch*2, out_ch, kernel),
                nn.ReLU(),
                nn.MaxPool1d(kernel, stride=kernel),
                )
        self.convs = [conv_layer2(max_length, 50, x) for x in [3,5]]
        self.convs += [conv_layer(max_length, 50, x) for x in [7,10]]

        self.conv_fc = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(500, 2*H),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
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
        lengths = Variable(torch.from_numpy(lengths))

        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers*2, data_in_torch.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers*2, data_in_torch.size(0), self.hidden_size))

        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if self.use_gpu:
            data_in_torch = data_in_torch.cuda()
            lengths = lengths.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()

        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)

        conv = torch.cat(tuple([conv(embeds) for conv in self.convs]), dim=2)
        conv = conv.view(batch_size, -1)
        conv = self.conv_fc(conv)

        embeds = torch.transpose(embeds, 1, 0)
        output, (hn, _) = self.lstm(embeds, (h0, c0))
        last_h = torch.cat((hn[-2], hn[-1]), dim=1)
        output = torch.transpose(output, 1, 0)
        mask = (data_in_torch > 0).unsqueeze(-1).float()
        output = output * mask

        if self.model_type == 'last':
            hidden = last_h
        elif self.model_type == 'max':
            hidden, idx = torch.max(output, 1)
        elif self.model_type == 'attn':
            attn = output.view(-1, self.max_length*2*self.embedding_dim)
            attn = self.attn(attn)
            attn = attn.unsqueeze(1)
            context = torch.matmul(attn, output)
            hidden = context.squeeze(1)
        else:
            # https://github.com/wballard/mailscanner/blob/attention/mailscanner/layers/attention.py
            attn_mat = self.attention_matrix(output)
            attn_vec = self.attention_vector(attn_mat.view(batch_size, -1))
            attn_vec = attn_vec.unsqueeze(2).expand(batch_size, self.max_length, self.rnn_dim)
            context = output * attn_vec
            hidden = torch.sum(context, dim=1)

        hidden = torch.cat((hidden, conv), dim=1)

        # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
        output = torch.sigmoid(self.fc(hidden)) * 10 + 0.5
        return output
