#!/usr/bin/env python3

import torch
from torch import nn
from tqdm.notebook import tqdm
tqdm.pandas()

"""input size (N x C x L), for example, the initial text is (batch_size, 70, 501)"""


def conv3(in_features, out_features, stride=1, padding=1, dilation=1, groups=1):
    """(1D-)Convolution with kernel size 3, with padding
    Args:
        in_features: nb input channels,
        out_features: nb output channels.
    """
    return nn.Conv1d(in_features, out_features, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=True)


def conv1(in_features, out_features, stride=1):
    """(1D-)Convolution with kernel size 1"""
    return nn.Conv1d(in_features, out_features, kernel_size=1, stride=stride, bias=True)


class CharResCNN_pre(nn.Module):
    def __init__(self, out_channels=170, nb_feats=70):
        """Almost the same as CharResCNN, but the output dimension is >1 (not yet logits)"""
        super().__init__()
        # self.out_dim = out_dim
        self.out_channels = out_channels
        self.relu = nn.ReLU()

        # use large kernel sizes to capture interaction between characters far away, e.g long words
        self.conv1 = nn.Sequential(
            nn.Conv1d(nb_feats, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.pool2 = nn.MaxPool1d(3)

        self.res3 = nn.Sequential(
            conv3(256, 256),
            nn.BatchNorm1d(256)
        )
        self.res4 = nn.Sequential(
            conv3(256, 256),
            nn.BatchNorm1d(256)
        )
        self.conv5 = nn.Sequential(
            conv3(256, 512, stride=3),
            nn.BatchNorm1d(512)
        )
        self.pool5 = nn.MaxPool1d(3)
        # After ResConv, shape = [B, C, L] = [B, 256, 18x3] B:Batch size
        """
        Input can be of size T x B x * where T is the length of the longest sequence (equal to lengths[0]), 
        B is the batch size, and * is any number of dimensions (including 0). 
        If batch_first is True, B x T x * input is expected.
        """
        self.rnn = nn.GRU(input_size=512,
                          hidden_size=50,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.5)
        """
        GRU input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence.
        """
        self.conv6 = nn.Sequential(
            conv3(100, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        """
        tweet = length of 501
        one-hot encoding = [70, 501]
        into CNN [B, C, H, W] ---> 1D conv: [B, C, L] ---> [64, 70, 501] ---> conv 3x3, do it on the last dim (length)
          ---> if kernel is 3x3, actually is [70, 3, 3], number of kernel is the output channel
        """
        # (batch, seqlen, nbfeats) -> (batch, nbfeats, seqlen) to feed into Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        identity3 = x
        x = self.res3(x)
        x += identity3
        x = self.relu(x)

        identity4 = x
        x = self.res4(x)
        x += identity4
        x = self.relu(x)

        # identity5 = x
        # x = self.res5(x)
        # x += identity5
        # x = self.relu(x)
        x = self.conv5(x)
        x = self.pool5(x)

        # x: (batch, channels, seqlen)
        x = x.permute(0, 2, 1)
        # x: (batch, seqlen, channels) = (batch, 27, 512)
        x, _ = self.rnn(x)
        # # hidden = [n layers * n directions, batch size, emb dim]
        # hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # x: (batch, seqlen, hiddenstates) = (batch, 27, 100)
        x = x.permute(0, 2, 1)

        # identity6 = x
        # x = self.res6(x)
        # x += identity6
        # x = self.relu(x)
        x = self.conv6(x)

        x = x.permute(0, 2, 1)
        # x: (batch, out_len, out_channels) = (batch, 18, out_channels) when input_len=501, the default
        return x


"""input size (N x C x L), for example, the initial text is (batch_size, 200, 131)"""


class WordCNN_pre(nn.Module):
    def __init__(self, out_channels=150, input_feats=200, input_len=131):
        """Almost the same as WordCNN, but the output dimension is >1 (not yet logits)

        Args:
            input_feats (int): the expected number of features of the input sequences
            input_len (int): the expected length of the input sequences (already padded)
        """
        super().__init__()
        # self.out_dim = out_dim
        self.out_channels = out_channels

        # transform the glove embeddings point-by-point
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_feats, 900, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(900, 250, 1),
            nn.ReLU(),
            nn.BatchNorm1d(250)
        )

        self.rnn = nn.GRU(input_size=250,
                          hidden_size=50,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.5)

        self.conv3 = nn.Sequential(
            nn.Conv1d(100, 120, 4),  # nb of input channels is 2*hidden_size of rnn (2* because bidirectional)
            nn.ReLU()
        )
        self.downsample3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv4 = nn.Sequential(
            nn.Conv1d(120, 140, 4),
            nn.ReLU(),
            nn.BatchNorm1d(140)
        )
        self.downsample4 = nn.MaxPool1d(kernel_size=3, stride=3)

        # transform the output timeslot-wise
        self.conv5 = nn.Sequential(
            nn.Conv1d(140, 500, 1),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(500, out_channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        # (batch, seqlen, nbfeats) -> (batch, nbfeats, seqlen) to feed into Conv1d
        x = x.permute(0, 2, 1)
        # x: (batch, input_channels, input_len) = (batch, 200, 131)
        x = self.conv1(x)
        x = self.conv2(x)
        # x: (batch, 250, input_len)

        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)  # don't need the final hidden state
        # x: (batch, seq_len, num_directions * hidden_size)
        x = x.permute(0, 2, 1)
        # x: (batch, num_directions * hidden_size, seq_len) = (batch, 100, 131)

        x = self.conv3(x)
        x = self.downsample3(x)
        x = self.conv4(x)
        x = self.downsample4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.permute(0, 2, 1)
        # x: (batch, out_len, out_channels) = (batch, 13, out_channels) when input_len=131, the default
        return x


"""input size (N x C x L), for example, the initial text is (batch_size, 200, 131)"""


class CharAndWordCNN(nn.Module):
    def __init__(self, charCNN_out_channels=60, wordCNN_out_channels=50, raw_nb_feats=70, pro_input_feats=200,
                 pro_input_len=131):
        """Almost the same as WordCNN, but the output dimension is >1 (not yet logits)

        Args:
            TODO
        """
        super().__init__()
        self.charCNN_pre = CharResCNN_pre(out_channels=charCNN_out_channels, nb_feats=raw_nb_feats)
        self.wordCNN_pre = WordCNN_pre(out_channels=wordCNN_out_channels, input_feats=pro_input_feats,
                                       input_len=pro_input_len)

        charCNN_out_dim = 1080  # determined on an ad hoc basis (computable by going through all the kernel sizes and strides but meh)
        wordCNN_out_dim = 650

        self.char_tofeats = nn.Sequential(  # convert to a single column of features (non temporal)
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(charCNN_out_dim, 300),
            nn.ReLU()
        )
        self.word_tofeats = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(wordCNN_out_dim, 300),
            nn.ReLU()
        )
        self.fc_char = nn.Sequential(
            nn.Linear(300, 1)
        )

        self.fc_word = nn.Sequential(
            nn.Linear(300, 1)
        )
        self.fc_combine = nn.Sequential(
            nn.Linear(300 + 300, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        # x_raw, x_pro = x
        x_char, x_word = x

        x_char = self.charCNN_pre(x_char)
        x_char = self.char_tofeats(x_char)  # (batch, 300)

        x_word = self.wordCNN_pre(x_word)
        x_word = self.word_tofeats(x_word)  # (batch, 300)

        output_char = self.fc_char(x_char)
        output_word = self.fc_word(x_word)

        x = torch.cat((x_char, x_word), dim=1)

        output = self.fc_combine(x)
        # x: (batch, 1)
        return output_char, output_word, output