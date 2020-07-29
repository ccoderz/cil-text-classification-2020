#! /bin/bash

mkdir stanford_glove_preprocessed
ruby -n preprocess-twitter.rb < twitter-datasets/test_data.txt > stanford_glove_preprocessed/test_data.txt
ruby -n preprocess-twitter.rb < twitter-datasets/train_pos.txt > stanford_glove_preprocessed/train_pos.txt
ruby -n preprocess-twitter.rb < twitter-datasets/train_neg.txt > stanford_glove_preprocessed/train_neg.txt
ruby -n preprocess-twitter.rb < twitter-datasets/train_pos_full.txt > stanford_glove_preprocessed/train_pos_full.txt
ruby -n preprocess-twitter.rb < twitter-datasets/train_neg_full.txt > stanford_glove_preprocessed/train_neg_full.txt

