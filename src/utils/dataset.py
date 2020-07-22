#!/usr/bin/env python3
import csv
import json
import torch
from torch.utils.data import Dataset

"""PyTorch dataset wrapping class"""


class TweetsDataset(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0=501):
        """Create Tweets News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase=True):
        self.label = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                self.label.append(int(row[1]))
                txt = ' '.join(row[0:-1])
                if lowercase:
                    txt = txt.lower()
                self.data.append(txt)

        self.y = torch.FloatTensor(self.label)

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples / float(self.label.count(c)) for c in label_set]
        return class_weight, num_class


class PredDataset(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0=501):
        """Create Tweets dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        self.l0 = l0
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        return X

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase=True):
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                txt = ' '.join(row)
                if lowercase:
                    txt = txt.lower()
                self.data.append(txt)

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)