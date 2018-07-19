import numpy as np
import datetime, time
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

quora_corpus = np.load("quora_corpus_int5.npy")
labels = np.load("quora_labels.npy")

w2v_embedding = {}
w2v_embedding['pivots100_7m'] = np.load("w2v_pivots100_7m.npy")
w2v_embedding['pivots100_6m'] = np.load("w2v_pivots100_6m.npy")
w2v_embedding['pivots100_5m'] = np.load("w2v_pivots100_5m.npy")
w2v_embedding['pivots100_4m'] = np.load("w2v_pivots100_4m.npy")
w2v_embedding['pivots100_3m'] = np.load("w2v_pivots100_3m.npy")
w2v_embedding['pivots100_2m'] = np.load("w2v_pivots100_2m.npy")
w2v_embedding['pivots100_1m'] = np.load("w2v_pivots100_1m.npy")
w2v_embedding['pivots100_05m'] = np.load("w2v_pivots100_05m.npy")
w2v_embedding['pivots100_01m'] = np.load("w2v_pivots100_01m.npy")
w2v_embedding['pivots100_005m'] = np.load("w2v_pivots100_005m.npy")
w2v_embedding['pivots100_001m'] = np.load("w2v_pivots100_001m.npy")

# separate question1 and question2
question1 = []
question2 = []
for n in range(int(len(quora_corpus) / 2)):
    question1.append(quora_corpus[2 * n])
    question2.append(quora_corpus[2 * n + 1])

q1_data = pad_sequences(question1, maxlen=25)
q2_data = pad_sequences(question2, maxlen=25)

# hyperparameter setup
max_sentence_len = 25
embed_dim = 50
dropout_rate = 0.1
vocab_size = len(w2v_embedding['pivots100_7m'])

# split cross validation set and test set
questions = np.stack((q1_data, q2_data), axis=1)
X_train, X_test, y_train, y_test = train_test_split(questions, labels, test_size=0.1, random_state=2018)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]


def Max_BoE(word_embedding):
    question1 = Input(shape=(max_sentence_len,))
    question2 = Input(shape=(max_sentence_len,))

    q1 = Embedding(input_dim=vocab_size,
                   output_dim=embed_dim,
                   weights=[word_embedding],
                   input_length=max_sentence_len,
                   trainable=False)(question1)
    q1 = TimeDistributed(Dense(embed_dim, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embed_dim,))(q1)

    q2 = Embedding(input_dim=vocab_size,
                   output_dim=embed_dim,
                   weights=[word_embedding],
                   input_length=max_sentence_len,
                   trainable=False)(question2)
    q2 = TimeDistributed(Dense(embed_dim, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embed_dim,))(q2)

    merged = concatenate([q1, q2])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(dropout_rate)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(dropout_rate)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(dropout_rate)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(dropout_rate)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

n_epoch = 50
val_split = 0.1
batch_size = 32

save_data = {}
for embed_name in w2v_embedding.keys():
    current_embed = w2v_embedding[embed_name]
    model = Max_BoE(current_embed)
    MODEL_WEIGHTS_FILE = 'Max_BOE_weights/'+embed_name+'_weights.h5'
    print('current embedding: ',embed_name)
    print("Starting training at", datetime.datetime.now())
    t0 = time.time()
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
    model_history = model.fit([Q1_train, Q2_train],
                        y_train,
                        epochs=n_epoch,
                        validation_split=val_split,
                        verbose=1,
                        batch_size=batch_size,
                        callbacks=callbacks)
    save_data[embed_name] = model_history.history
    t1 = time.time()
    print("Training ended at", datetime.datetime.now())
    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")

f = open('pivots100_data.txt','w')
f.write(str(save_data))
f.close()