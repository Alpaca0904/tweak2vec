import numpy as np
import tensorflow as tf
import random
from collections import Counter
import datetime, time, json


def create_word_pairs(int_corpus, window_size):
    idx_pairs = []
    # for each snetence
    for sentence in int_corpus:
        # for each center word
        for center_word_pos in range(len(sentence)):
            center_word_idx = sentence[center_word_pos]
            # for each context word within window
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(sentence) or center_word_pos == context_word_pos:
                    continue
                context_word_idx = sentence[context_word_pos]
                idx_pairs.append((center_word_idx, context_word_idx))

    return idx_pairs

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

# create word pairs
corpus = np.load("/Users/zhang/MscProject_tweak2vec/corpus/quora_corpus_int5.npy").tolist()
idx_pairs_SG = create_word_pairs(corpus, window_size = 5)
print('totally {0} word pairs'.format(len(idx_pairs_SG)))

wordlist = np.load('/Users/zhang/MscProject_tweak2vec/corpus/quora_vocab5.npy').tolist()
wordlist.append(['UNK','0'])
word2idx = {w[0]: wordlist.index(w) for w in wordlist }
idx2word = {wordlist.index(w): w[0] for w in wordlist }

#load pivot word vectors
f = open('pivots_google_10000.txt','r')
a = f.read()
pivots_vec = eval(a)
f.close()
len(pivots_vec)
update_idx = []
update_vec = []
for i in pivots_vec.keys():
    update_idx.append(i)
    update_vec.append(pivots_vec[i])
print('load {0} pivot words:{1}'.format(len(update_idx),[idx2word[i] for i in update_idx]))

# build graph
n_vocab = len(word2idx)
n_embedding = 50
reg_constant = 0.01
n_sampled = 100
learning_rate = 0.001
epochs = 20
batch_size = 1000 # number of samples each iteration

train_graph = tf.Graph()
with train_graph.as_default():
    # input layer
    inputs = tf.placeholder(tf.int32, [batch_size], name='inputs')
    # labels is 2 dimensional as required by tf.nn.sampled_softmax_loss used for negative sampling.
    labels = tf.placeholder(tf.int32, [None, None], name='labels')

    # embedding layer
    init_width = 0.5 / n_embedding
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -init_width, init_width))
    embed = tf.nn.embedding_lookup(embedding, inputs)

    #add regularization term
    reg_mat = []
    inputs_unpacked = tf.unstack(inputs)
    for i in inputs_unpacked:
        if i in pivots_vec.keys():
            reg_mat.append((embedding[i] - np.array(pivots_vec[i])) ** 2)
    reg_loss = reg_constant * tf.reduce_sum(reg_mat)

    # sampled softmax layer
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding)), name="softmax_weights")
    softmax_b = tf.Variable(tf.zeros(n_vocab), name="softmax_bias")
    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(
        weights=softmax_w,
        biases=softmax_b,
        labels=labels,
        inputs=embed,
        num_sampled=n_sampled,
        num_classes=n_vocab)
    cost = tf.reduce_mean(loss)

    total_cost = cost + reg_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)

# start training
print("Starting training at", datetime.datetime.now())
t0 = time.time()

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    loss_best = 100
    loss_list = []
    iteration_best = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(idx_pairs_SG, batch_size)
        start = time.time()
        for x, y in batches:
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([total_cost, optimizer], feed_dict=feed)

            loss += train_loss

            if loss < loss_best:
                W = sess.run(embedding).tolist()
                iteration_best = iteration
                loss_best = loss

            if iteration % 1000 == 0:
                end = time.time()
                loss_list.append(loss / 1000)
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 1000),
                      "{:.4f} sec/batch".format((end - start) / 1000))

                np.save('w2v_pivots100_50d.npy', np.array(W))
                np.save('loss_pivots100_50d.npy', np.array(loss_list))

                loss = 0
                start = time.time()
            iteration += 1


# save embedding matrics
np.save('w2v_pivots100_50d.npy',np.array(W))
np.save('loss_pivots100_50d.npy',np.array(loss_list))

print('best result at iteration:{0}'.format(iteration_best))