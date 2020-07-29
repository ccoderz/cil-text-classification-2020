#!/usr/bin/env python3
import os
import csv
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import GloVe

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


class TweetsAsCharsAndWordsDataset(Dataset):
    def __init__(self, data_path, alphabet_path, is_labeled=True, l0=501, l1=131, max_samples=None,
                 word_emb_name="twitter.27B", word_emb_dim=200,
                 vector_cache_path=None):
        """A dataset object whose samples consist of *both*
            - the (padded) concatenation of the word vectors of a tweet, and
            - the per-character one-hot encoding of the same tweet.

        Arguments:
            data_path: path of (label and) data file in csv.
            alphabet_path: path of alphabet json file.
            is_labeled: whether the data_path file contains labels, or only the tweets.
            l1: max length of a sample, in nb of characters.
            l1: max length of a sample, in nb of words.
            max_samples: (for dev,) only keep the max_samples first samples of the data.

            word_emb_name: name of the word embedding to use, used by torchtext.GloVe.
            word_emb_dim: dimension of the word embedding to use, used by torchtext.GloVe.
            vector_cache_path: path to cache directory, used by torchtext.GloVe.
        """
        self.glove = GloVe(name=word_emb_name, dim=word_emb_dim, cache=vector_cache_path)
        print("loaded pretrained GloVe word-embeddings.")
        self.data_path = data_path
        self.alphabet_path = alphabet_path
        self.is_labeled = is_labeled
        self.l0 = l0
        self.l1 = l1
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))
        self.raw_nb_feats = len(self.alphabet)
        self.pro_nb_feats = word_emb_dim
        # TODO: setting max_samples only makes sense if the csv itself was shuffled
        # X_txt = pd.read_csv(data_path, nrows=max_samples) # only keep max_samples first samples, or keep all if None
        X_txt = pd.read_csv(data_path)
        if max_samples:
            assert is_labeled, "must not use `max_samples` for unlabeled (assumed test-) data, as shuffling would modify the samples' ordering"
            X_txt = X_txt.sample(frac=1).reset_index(drop=True).iloc[
                    :max_samples]  # shuffle then select max_samples first
        self.y = X_txt['label'].to_numpy().astype(np.integer, copy=False) if is_labeled else None
        self.X_pro = X_txt['preprocessed_segmented_tweet'].to_numpy()
        self.X_raw = X_txt['raw_tweet'].to_numpy()

    def __len__(self):
        return self.X_raw.shape[0]

    def __getitem__(self, idx):
        X_raw = self.get_item_raw(idx)
        X_pro = self.get_item_pro(idx)
        # even if X consists of two distinct parts, still output X,y so that auxiliary functions work without modification
        if self.is_labeled:
            return (X_raw, X_pro), self.y[idx]
        else:
            return (X_raw, X_pro)

    def get_item_pro(self, idx):
        words = self.X_pro[idx].lower().split()
        words += [""] * (self.l1 - len(words))  # pad with zeros until of correct size
        assert len(words) == self.l1
        X = self.glove.get_vecs_by_tokens(words, lower_case_backup=True)
        # for i in np.where(~X.bool().all(axis=1))[0]: # print OOV words
        #     if words[i] != "":
        #         print("out-of-vocabulary:", i, words[i])
        assert X.shape == (self.l1, self.glove.dim)
        return X

    def get_item_raw(self, idx):
        seq = self.X_raw[idx]
        X = self.oneHotEncode(seq)
        assert X.shape == (self.l0, self.raw_nb_feats)  # NOTE: this is the transpose of what Xiaochen did
        return X

    def char2idx(self, character):
        return self.alphabet.find(character)

    def oneHotEncode(self, seq):
        X = torch.zeros(self.l0, self.raw_nb_feats)
        for i, char in enumerate(seq[::-1]):
            char_idx = self.char2idx(char)
            if char_idx != -1:  # if char is in present in self.alphabet
                X[i, char_idx] = 1.0
        return X