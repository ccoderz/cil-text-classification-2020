- preprocess-twitter.rb: slightly modified version of the Ruby script fetched on [GloVe's webpage](https://nlp.stanford.edu/projects/glove/), where we also downloaded the GloVe pretrained embedding
- run_script.sh: a bash script to run the Ruby script
- longest_tweet.py: a tiny utility script to determine the longest tweet length

- the glove embeddings cache, in aux-data, are downloaded automatically by `torchtext.vocab.GloVe` when running the `tran_clear.py`