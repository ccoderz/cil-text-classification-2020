#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

"""input size (N x C x L), for example, the initial text is (batch_size, 70, 501)"""


def conv3x3(in_features, out_features, stride=1, padding=1, dilation=1, groups=1):
    """3x3 convolution with padding
    args:
    in_features: input channels,
    out_features: output channels.
    """
    return nn.Conv1d(in_features, out_features, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=True)


def conv1x1(in_features, out_features, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_features, out_features, kernel_size=1, stride=stride, bias=True)


class CSRRes4d(nn.Module):
    def __init__(self, args):
        super(CSRRes4d, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.num_features, 256, kernel_size=7, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.downsample = nn.Sequential(
            conv1x1(256, 512, stride=2),
            nn.BatchNorm1d(512)
        )

        self.res2 = nn.Sequential(
            conv3x3(256, 256),
            nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),

            # conv3x3(256, 256),
            # nn.BatchNorm1d(256)
        )

        self.res3 = nn.Sequential(
            conv3x3(256, 256),
            nn.BatchNorm1d(256),
            # nn.ReLU(inplace=True),

            # conv3x3(256, 256),
            # nn.BatchNorm1d(256)
        )

        self.res4 = nn.Sequential(
            conv3x3(256, 512, stride=2),
            nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),

            # conv3x3(256, 256),
            # nn.BatchNorm1d(256)
        )

        self.res5 = nn.Sequential(
            conv3x3(512, 512),
            nn.BatchNorm1d(512),
            # nn.ReLU(inplace=True),

            # conv3x3(512, 512),
            # nn.BatchNorm1d(512)
        )
        # After ResConv, shape = [B, C, L] = [B, 256, 18x3] B:Batch size

        """
        Input can be of size T x B x * where T is the length of the longest sequence (equal to lengths[0]), 
        B is the batch size, and * is any number of dimensions (including 0). 
        If batch_first is True, B x T x * input is expected.
        """

        self.rnn = nn.GRU(input_size=512,
                          hidden_size=256,
                          num_layers=2,
                          bidirectional=True,
                          batch_first=True,
                          dropout=0.5)

        """
        GRU input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence.
        """
        # self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.out1 = nn.Linear(1024, 512)
        self.out2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        tweet = length of 501
        one-hot encoding = [70, 501]
        into CNN [B, C, H, W] ---> 1D conv: [B, C, L] ---> [64, 70, 501] ---> conv 3x3, do it on the last dim (length)
        ---> if kernel is 3x3, actually is [70, 3, 3], number of kernel is the output channel

        """
        x = self.conv1(x)
        # x = self.conv2(x)

        identity2 = x
        x = self.res2(x)
        x += identity2
        x = self.relu(x)

        identity3 = x
        x = self.res3(x)
        x += identity3
        x = self.relu(x)

        identity4 = self.downsample(x)
        x = self.res4(x)
        x += identity4
        x = self.relu(x)

        identity5 = x
        x = self.res5(x)
        x += identity5
        x = self.relu(x)

        x = self.maxpool1d(x)
        # x = [B, 512, 18]
        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, emb dim]
        _, hidden = self.rnn(x)
        # hidden = [n layers * n directions, batch size, emb dim]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # output = self.out1(hidden)
        output = self.out2(hidden)
        # output = [batch size, out dim]

        return output