import numpy as np
import tensorflow as tf
import random
from collections import Counter
import datetime, time, json


def create_word_pairs(int_corpus, window_size, stop_size):
    idx_pairs = []
    tokens = 0
    # for each snetence
    for sentence in int_corpus:
        # for each center word
        for center_word_pos in range(len(sentence)):
            center_word_idx = sentence[center_word_pos]
            tokens += 1
            if tokens >= stop_size:
                return idx_pairs, tokens
            else:
                # for each context word within window
                for w in range(-window_size, window_size + 1):
                    context_word_pos = center_word_pos + w
                    # make soure not jump out sentence
                    if context_word_pos < 0 or context_word_pos >= len(sentence) or center_word_pos == context_word_pos:
                        continue
                    context_word_idx = sentence[context_word_pos]
                    idx_pairs.append((center_word_idx, context_word_idx))

    return idx_pairs, tokens

def get_batches(idx_pairs, batch_size):
    n_batches = len(idx_pairs) // batch_size
    idx_pairs = idx_pairs[:n_batches*batch_size]
    for idx in range(0, len(idx_pairs), batch_size):
        x, y = [], []
        batch = idx_pairs[idx:idx+batch_size]
        for ii in range (len(batch)):
            x.append(batch[ii][0])
            y.append(batch[ii][1])
        yield x, y

corpus = np.load("/Users/zhang/MscProject_tweak2vec/corpus/quora_corpus_int5.npy").tolist()

corpus_shuffle = corpus[:]

random.shuffle(corpus_shuffle)
idx_pairs_SG_7m, tokens = create_word_pairs(corpus_shuffle, window_size = 5, stop_size = 7000000)
print('totally {0} word pairs'.format(len(idx_pairs_SG_7m)))
print('totally {0} tokens'.format(tokens))

