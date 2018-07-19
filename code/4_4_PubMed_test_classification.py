import numpy as np
import tensorflow as tf
import datetime, time

def zeros_padding(lst):
    inner_mean_len = 100
    result = np.zeros([len(lst), inner_mean_len])
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            if j < inner_mean_len:
                result[i][j] = val
    return result


def docs_to_lines(docs):
    l = []
    for doc in docs:
        for line in doc:
            l.append(line)
    return l


def to_one_of_k(int_targets, num_classes):
    one_of_k_targets = np.zeros((np.array(int_targets).shape[0], num_classes))
    one_of_k_targets[range(np.array(int_targets).shape[0]), int_targets] = 1
    return one_of_k_targets


def get_batches(input_x, input_y, batch_size, isShuffle=False):
    n_batches = len(input_x) // batch_size
    train_size = n_batches * batch_size
    if isShuffle:
        shuffle_idx = np.random.permutation(np.arange(len(input_x)))
        train_x = input_x[shuffle_idx]
        train_y = np.array(input_y)[shuffle_idx]
    else:
        train_x = input_x[:train_size]
        train_y = np.array(input_y)[:train_size]
    for idx in range(0, len(train_x), batch_size):
        x = train_x[idx:idx + batch_size]
        y = train_y[idx:idx + batch_size]
        yy = to_one_of_k(y.astype(np.int32), 5)
        yield x, yy

embedding_pubmed = {}
embedding_pubmed['pivots1000_2m'] = np.load('w2v_pivots1000_2m.npy')
embedding_pubmed['pivots1000_1m'] = np.load('w2v_pivots1000_1m.npy')
embedding_pubmed['pivots1000_05m'] = np.load('w2v_pivots1000_05m.npy')
embedding_pubmed['pivots1000_01m'] = np.load('w2v_pivots1000_01m.npy')
embedding_pubmed['pivots1000_005m'] = np.load('w2v_pivots1000_005m.npy')
embedding_pubmed['pivots1000_001m'] = np.load('w2v_pivots1000_001m.npy')

embedding_dim = 50
seq_length = 100
num_classes = 5
num_filters = 256  # number of kernels
kernel_size = 5
vocab_size = len(embedding_pubmed['pivots1000_2m'])

hidden_dim = 128

keep_prob_rate = 0.75
learning_rate = 1e-3

batch_size = 100
num_epoch = 8

print_per_batch = 100
save_per_batch = 10

file_train_x = 'pubmed_train_x.npy'
file_train_y = 'pubmed_train_y.npy'
file_val_x = 'pubmed_dev_x.npy'
file_val_y = 'pubmed_dev_y.npy'

train_x = zeros_padding( docs_to_lines( np.load(file_train_x).tolist() ) )
train_y = docs_to_lines( np.load(file_train_y).tolist() )
val_x = zeros_padding( docs_to_lines( np.load(file_val_x).tolist() ) )
val_y = to_one_of_k(docs_to_lines( np.load(file_val_y).tolist() ),5)

train_graph = tf.Graph()
with train_graph.as_default():
    input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
    input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
    embedding = tf.placeholder(tf.float32, [vocab_size, embedding_dim], name='embedding')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope("embedding"):
        # embedding layer
        embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
    with tf.name_scope("CNN"):
        # CNN layer
        conv1 = tf.layers.conv1d(inputs=embedding_inputs, filters=num_filters,
                                 kernel_size=kernel_size, padding="VALID", activation=tf.nn.relu,
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(0.001), )
        # global maxpooling layer
        pool1 = tf.reduce_max(conv1, reduction_indices=[1])
        # pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=5, strides=5,padding="VALID")
        bn1 = tf.layers.batch_normalization(pool1)

        conv2 = tf.layers.conv1d(inputs=conv1, filters=num_filters,
                                 kernel_size=kernel_size, padding="VALID", activation=tf.nn.relu,
                                 activity_regularizer=tf.contrib.layers.l2_regularizer(0.001), )
        pool2 = tf.reduce_max(conv2, reduction_indices=[1])
        bn2 = tf.layers.batch_normalization(pool2)

    with tf.name_scope("classifier"):
        fc = tf.layers.dense(bn2, hidden_dim, name='fc1')
        fc = tf.contrib.layers.dropout(fc, keep_prob)
        fc = tf.nn.relu(fc)
        # classifier
        logits = tf.layers.dense(fc, num_classes, name='fc2')
        y_pred_class = tf.argmax(tf.nn.softmax(logits), 1)
    with tf.name_scope("optimize"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred_class)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# for embed_type in embedding_pubmed.keys():
for embed_type in embedding_pubmed.keys():

    print(embed_type + " Starting training at", datetime.datetime.now())

    train_loss_lst = []
    train_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        t_loss = 0
        t_acc = 0
        v_best_acc = 0
        iteration = 0
        train_batches_size = len(train_x) // batch_size
        val_batches_size = len(val_x) // batch_size

        for e in range(1, num_epoch + 1):
            train_batches = get_batches(train_x, train_y, batch_size, isShuffle=True)
            val_batches = get_batches(val_x, val_y, batch_size)

            start = time.time()
            # training
            for train_inputs, train_targets in train_batches:
                iteration += 1
                feed = {input_x: train_inputs,
                        input_y: train_targets,
                        embedding: embedding_pubmed[embed_type],
                        keep_prob: keep_prob_rate}
                train_loss, train_acc, _ = sess.run([loss, accuracy, optimizer], feed_dict=feed)
                t_loss += train_loss
                t_acc += train_acc
                if iteration % 600 == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, num_epoch),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(t_loss / 600),
                          "Avg. Training acc: {:.4f}".format(t_acc / 600),
                          "{:.4f} sec/batch".format((end - start) / 600))
                    t_loss = 0
                    t_acc = 0
                    start = time.time()
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)

            # validation
            feed = {input_x: val_x,
                    input_y: val_y,
                    embedding: embedding_pubmed[embed_type],
                    keep_prob: 1}
            val_loss, val_acc, y_pred = sess.run([loss, accuracy, y_pred_class], feed_dict=feed)
            if val_acc > v_best_acc:
                v_best_acc = val_acc
                y_predict = y_pred

            end = time.time()
            print("Epoch {}/{}".format(e, num_epoch),
                  "Avg. Val. loss: {:.4f}".format(val_loss),
                  "Avg. Val. acc: {:.4f}".format(val_acc),
                  "{:.4f} sec".format(end - start),
                  "--------------------------------")
            val_loss_lst.append(val_loss)
            val_acc_lst.append(val_acc)
        y_label = docs_to_lines(np.load(file_val_y).tolist())
        confusion_mat = tf.confusion_matrix(y_label, y_predict, 5)
        confusion = sess.run(confusion_mat)

    print("Finish at", datetime.datetime.now())

    np.save(embed_type + '_train_acc.npy', np.array(train_acc_lst))
    np.save(embed_type + '_train_loss.npy', np.array(train_loss_lst))
    np.save(embed_type + '_val_acc.npy', np.array(val_acc_lst))
    np.save(embed_type + '_val_loss.npy', np.array(val_loss_lst))
    np.save(embed_type + '_confusion.npy', confusion)