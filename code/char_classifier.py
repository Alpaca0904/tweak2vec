#classifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
import os,sys

from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
#------------------------------------------------------#
# Functions to preprocess tweet texts
#------------------------------------------------------#
def remove_urls(s):
    url_regex = re.compile(
    r'(?:http|ftp)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
    r'\[?[A-F0-9]*:[A-F0-9:]+\]?)/'  # ...or ipv6
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    s = re.sub(url_regex,'',s)
    return s

class tweet_data(Dataset):
    '''data generator for classification task'''
    def __init__(self,dataframe):
        self.total = dataframe.shape[0]
        self.data = dataframe
        self.target_col = dataframe.columns[1]
        self.target = dataframe[self.target_col] #panda series
        self.text_col = dataframe.columns[-1]
        self.text = dataframe[self.text_col]
        self.tknzer = TweetTokenizer()
        return
    def __len__(self):
        return self.total

    def __getitem__(self,index):
        '''generate one sample of data'''
        target =  str(self.target[index])
        sent = self.text[index].lower()
        sent = self.clean_tweet(sent) #remove urls,hashtags, users
        sent = self.tknzer.tokenize(sent)
        return sent,target

    def preprocess(self,sent):
        '''remove symbols,and underscores(non-alpha)
        and tokenize
        remove digits'''
        reg = re.compile('([^\s\w]|_)+')
        sent = re.sub(reg,'',sent.lower())
        return sent
    def clean_tweet(self,s):
        '''s: untokenized string'''
        s,_ = self.extract_remove_hashtags(s)
        s = self.remove_users(s)
        s = self.simple_remove_urls(s)
        return s

    def extract_remove_hashtags(self,s):
        '''s:untokenized and lowercased sent string'''
        tknzer = TweetTokenizer()
        tags = list(set(part[1:] for part in tknzer.tokenize(s) if part.startswith('#')))
        for i in tags:
            s = s.replace('#'+i,'')
        return s,tags
    def remove_users(self,s):
        '''s:untokenized and lowercased sent string'''
        tknzer = TweetTokenizer()
        users = list(set(part[1:] for part in tknzer.tokenize(s) if part.startswith('@')))
        for i in users:
            s = s.replace('@'+i,'')
        return s
    def simple_remove_urls(self,s):
        tknzer = TweetTokenizer()
        '''s:untokenized and lowercased sent string'''
        urls = list(set(part for part in tknzer.tokenize(s) if (part.startswith('http') or part.startswith('ftp'))))
        for i in urls:
            s = s.replace(i,'')
        return s


class char_lstm(nn.Module):
    '''forward output: output sequence of the last bilstm layer
        size = batch*longest_seq*(2hidden_size)'''
    def __init__(self,char_vocab_size,char_embed=300,hidden = 150,char_layer = 2):
        super().__init__()
        self.input_size = char_embed
        self.hidden_size = hidden
        self.layers = char_layer
        self.char_embeddings = nn.Embedding(char_vocab_size,self.input_size,padding_idx=0)
        self.bilstm = nn.LSTM(self.input_size,self.hidden_size,self.layers,bidirectional=True,dropout=0.5)
    def forward(self,char_ids,word_lens,init_hidden):
        char_embeds = self.char_embeddings(char_ids)
        #packed_padded_sequence,input of size BxTx*
        X = torch.nn.utils.rnn.pack_padded_sequence(char_embeds,word_lens,batch_first = True)
        hidden,cell = self.bilstm(X,init_hidden)
        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
        return hidden,cell

    def init_hidden(self,batch_size,device):
        start_hidden = (torch.randn(self.layers*2, batch_size, self.hidden_size,device=device),
                                torch.randn(self.layers*2, batch_size, self.hidden_size,device=device)) #h0,c0
        return start_hidden
class classifier(nn.Module):
    def __init__(self,output_classes,input_size=300,hidden_size=100):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden_size
        self.out_classes = output_classes
        self.linear = nn.Linear(self.input_size,self.hidden)
        self.outlin = nn.Linear(self.hidden,self.out_classes)
        self.sent_vec = torch.Tensor()

    def forward(self,inputs):
        '''input:sequences of vectors,batch*seq*dim'''
        self.sent_vec = torch.mean(inputs,dim = 1) #batch*input_size
        out = torch.sigmoid(self.linear(self.sent_vec))
        out = self.outlin(out)
        log_probs = F.log_softmax(out,dim = -1) #for NLLLoss
        return log_probs

def obtain_sent_rep(words,embedding_hashtable):
    '''arg: words: a list of words in the preprocessed&tokenized sentence
                        e.g. "hello world" = ['hello','word']
            embedding_matrix = pre-trained word embeddings,size [vocabulary*INPUT_SIZE]
            embedding_hashtable = a dictionary maps word to its vector;
                                    with word2vec obtained with gensim, embedding matrix = model
       return: sentence vector of the same size as input word vectors'''
    projection_sent = [ embedding_hashtable[word] for word in words]
    return projection_sent
def label2index(label_series):
    labels = label_series.unique()
    labels = [str(i) for i in labels]
    num_classes = len(labels)
    print("Number of labels: %d" %(len(labels)))
    label2idx = dict(zip(labels,range(len(labels))))
    return label2idx,num_classes
def word2char_id(word,char2index):
    '''convert a string of word to a collection of chars;
    and map to char indices; remove unknow chars in a word
    use white space as the start/end marker of word'''
    batch_chars = [list(i) for i in word] # assume word=[seq_len*batch_size]
    char_ids = []
    index = []
    for idx,chars in enumerate(batch_chars):
        chars = [char2index[i] for i in chars if i in char2index]
        if len(chars) > 0:
            char_ids.append(chars)
            index.append(idx)
    return char_ids,index

def padding(indices):
    '''input: int lists of variable length sequence minibatch
        e.g. x= [[0,1,2,3,4,5],[7,7],[6,8]]
            =batch_size*seq_len
        Also, assume the padding token always has an index of 0'''
    #sort the batch in the descending order of length
    words_idx = range(len(indices))
    sorted_idx = sorted(words_idx, key=lambda idx:len(indices[idx]),reverse = True)

    indices = sorted(indices,key=lambda seq:len(seq),reverse=True)
    batch_size = len(indices)
    lengths = [len(seq) for seq in indices]
    longest_seq = max(lengths)
    pad_token = int(0)
    padded_indices = np.ones((batch_size,longest_seq),dtype = int)*pad_token

    for i,len_i in enumerate(lengths):
        seq = indices[i]
        padded_indices[i,0:len_i] = seq[0:len_i]

    return padded_indices,lengths,sorted_idx

def train(train_set,train_total,val_set,val_total,char2idx,label2idx,device,log_file,lr=0.01,epoches = 10):
    loss_fn = nn.NLLLoss().to(device)
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    train_f1 = []
    val_f1 = []
    epoch_loss = 0 #train_error per epoch
    best_val_loss = 0
    best_epoch = 0

    batch_size = 1 #batch size of sentence
    epoches = epoches

    start = time.time()
    optimizer = optim.Adam([
                {'params': char_model.parameters()},
                {'params': text_classifier.parameters(), 'lr': lr}
            ], lr=lr,weight_decay=0)
    # train_error,train_acc,train_f = evaluation(train_set,train_total,char_model,classifier,char2idx,label2idx,device)
    # train_loss.append(train_error)
    # train_accuracy.append(train_acc)
    # train_f1.append(train_f)
    # print("init train error: %0.4f" %(train_error))
    # print("init train accuracy: %0.4f" %(train_acc))
    #
    #
    # val_error,val_acc,val_f = evaluation(val_set,val_total,char_model,classifier,char2idx,label2idx,device)
    # val_loss.append(val_error)
    # val_accuracy.append(val_acc)
    # val_f1.append(val_f)
    # best_val_loss = val_error
    # best_val_accuracy = val_acc
    #
    # print("init best val error: %0.4f" %(val_error))
    # print("init best val accuracy: %0.4f" %(val_acc))
    # results_pd = {'train_loss':train_loss,'train_accuracy':train_accuracy,'train_f1':train_f1,
    #                 'val_loss':val_loss,'val_accuracy':val_accuracy,'val_f1':val_f1}
    #
    # with open('./char/results_pd.csv','w') as f:
    #         df = pd.DataFrame(results_pd)
    #         df.to_csv(f)

    for epoch in range(epoches):
        epoch_start = time.time()
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True)

        char_model.train()#slightly update char_model for classification task
        text_classifier.train()

        print("Epoch %d training..." %epoch)
        for batch_idx, data in enumerate(train_loader):
            sent = data[0] #size = seq*batch
            sent_len=len(sent)
            label = data[1]
            # print("sent \n")
            # print(sent)
            # print(type(sent))
            # print("label")
            # print(label)
            # print(type(label))

            optimizer.zero_grad()
            words = [sent[i][0] for i in range(len(sent))] #words in sent1,batch_size always= 1
            # print("words")
            #print(words)
            char_ids,index = word2char_id(words,char2idx)
            # print(char_ids)
            # print(index)
            if len(char_ids) == 0: #if empty
                continue

            #size filtered_batch*int
            label_idx= [label2idx[label[i]] for i in range(len(label))]
            #print(label_idx)
            label_idx = torch.tensor(label_idx,dtype=torch.long,device = device)

            char_ids,word_lens,sorted_idx = padding(char_ids) #padding,return np.array
            #print(char_ids)
            char_ids = torch.tensor(char_ids,dtype=torch.long,device = device)
            #feed into char_lstm
            output,_ = char_model(char_ids,word_lens,char_model.init_hidden(len(char_ids),device))
            fw_out = output[:,:,:char_model.hidden_size]
            bk_out = output[:,:,char_model.hidden_size:]
            #word vec dim = cat(batch*hidden,batch*hidden,dim = 1) = batch * 2hidden
            word_vec = torch.Tensor().to(device)
            for i,length in enumerate(word_lens):
                vec=torch.cat((fw_out[i,length-1,:],bk_out[i,0,:]),dim = -1)
                word_vec = torch.cat([word_vec,vec.view(1,-1)],dim=0)

            #print("word_vec.requires_grad")
            #print(word_vec.requires_grad)
            #reorder to original sentence
            sorted_word_vec = word_vec[sorted_idx]
            sent_rep = sorted_word_vec.view(1,sorted_word_vec.size(0),sorted_word_vec.size(1))#dim = batch*seq*dim
            #print("sorted_word_vec.requires_grad")
            #print(sorted_word_vec.requires_grad)

            log_probs = text_classifier(sent_rep)
            loss = loss_fn(log_probs,label_idx)

            loss.backward()
            optimizer.step()
            '''evaluation'''
            
        train_error,train_acc,train_f = evaluation(train_set,train_total,char_model,text_classifier,char2idx,label2idx,device)
        train_loss.append(train_error)
        train_accuracy.append(train_acc)
        train_f1.append(train_f)
        print("train error: %0.4f" %(train_error))
        print("train accuracy: %0.4f" %(train_acc))
        print("train f1: %0.4f" %(train_f))

        val_error,val_acc,val_f = evaluation(val_set,val_total,char_model,text_classifier,char2idx,label2idx,device)
        val_loss.append(val_error)
        val_accuracy.append(val_acc)
        val_f1.append(val_f)
        print("val error: %0.4f" %(val_error))
        print("val accuracy: %0.4f" %(val_acc))
        print("val f1: %0.4f" %(val_acc))

        #save best model
        if not best_val_loss or val_error < best_val_loss:
            with open("./char/best_char_model_for_classifier", 'wb') as f:
                torch.save(char_model.state_dict(), f)
            with open("./char/best_classifier",'wb') as f:
                torch.save(text_classifier.state_dict(), f)
            best_val_loss = val_error
            best_val_accuracy = val_acc
            best_epoch = epoch

        with open(log_file,'a') as f:
            f.write("Epoch %d : \n" %epoch)
            f.write("best epoch: %d \n" %best_epoch)
            f.write("best val loss: %0.4f, accuracy: %0.4f \n" %(best_val_loss,best_val_accuracy))
        results_pd = {'train_loss':train_loss,'train_accuracy':train_accuracy,'train_f1':train_f1,
                        'val_loss':val_loss,'val_accuracy':val_accuracy,'val_f1':val_f1}

        with open('./char/results_pd.csv','w') as f:
                df = pd.DataFrame(results_pd)
                df.to_csv(f)

        epoch_time = time.time() - epoch_start
        print("time: %0.2f" %(epoch_time))

    print("Finish training")
    return train_loss,val_loss,train_accuracy,val_accuracy,best_val_loss,best_epoch
def evaluation(data_set,total,char_model,classifier,char2idx,label2idx,device):
    loss_fn = nn.NLLLoss().to(device)
    total_loss = 0
    label_int = []
    prediction = []

    batch_size = 1 #batch size of sentence

    loader = DataLoader(data_set, batch_size=batch_size, num_workers=4, shuffle=False)
    char_model.eval()
    classifier.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            sent = data[0]
            label = data[1]
            sent_len=len(sent)

            words = [data[0][i][0] for i in range(len(sent))] #words in sent,batch_size always= 1
            char_ids,index = word2char_id(words,char2idx)
            if len(char_ids) == 0: #if empty    batch_size = 1 #batch size of sentence
                continue

            #size filtered_batch*int
            label_idx= [label2idx[label[i]] for i in range(len(label))]
            label_idx = torch.tensor(label_idx,dtype=torch.long,device = device)
            label_int.extend(label_idx)

            char_ids,word_lens,sorted_idx = padding(char_ids) #padding,return np.array
            char_ids = torch.tensor(char_ids,dtype=torch.long,device = device)

            output,_ = char_model(char_ids,word_lens,char_model.init_hidden(len(char_ids),device))
            fw_out = output[:,:,:char_model.hidden_size]
            bk_out = output[:,:,char_model.hidden_size:]
            #word vec dim = cat(batch*hidden,batch*hidden,dim = 1) = batch * 2hidden
            word_vec = torch.Tensor().to(device)
            for i,length in enumerate(word_lens):
                vec=torch.cat((fw_out[i,length-1,:],bk_out[i,0,:]),dim = -1)
                word_vec = torch.cat([word_vec,vec.view(1,-1)],dim=0)

            #reorder to original sentence
            sorted_word_vec = word_vec[sorted_idx] #seq*dim #avoid in-place operation
            sent_rep = sorted_word_vec.view(1,sorted_word_vec.size(0),sorted_word_vec.size(1))#dim = batch*seq*dim
            log_probs = classifier(sent_rep)

            pred_class = torch.argmax(log_probs, dim = 1)
            prediction.append(pred_class.item())

            loss = loss_fn(log_probs,label_idx)
            total_loss += torch.mul(loss,batch_size)

    accuracy = metrics.accuracy_score(label_int,prediction)
    f1 = metrics.f1_score(label_int,prediction,average = 'macro')
    return torch.div(total_loss,total),accuracy,f1
def plt_train_val_loss(train_loss,val_loss,train_accuracy,val_accuracy,plt_file="./char/classifier_error"):
    print("plot")
    plt.clf()
    plt.figure(1)
    x_epoch = np.arange(0,len(train_loss),dtype=int)
    plt.xlabel('epoch')
    plt.subplot(211)#num_row,num_column,subfig
    plt.plot(x_epoch, train_loss,'b^')
    plt.plot(x_epoch, val_loss,'r^')

    plt.subplot(212)
    plt.plot(x_epoch, train_accuracy,'bo')
    plt.plot(x_epoch, val_accuracy,'ro')

    file = plt_file
    plt.savefig(file)
    #plt.show()
    return

if __name__ == '__main__':
    torch.manual_seed(1)
    train_file = "data/tweet_sentiment_train.csv"
    val_file = "data/tweet_sentiment_val.csv"
    test_file = "data/tweet_sentiment_test.csv"

    train_df = pd.read_csv(train_file)
    train_df = train_df[0:300]#8839
    train_total = train_df.shape[0]

    val_df = pd.read_csv(val_file)
    val_df = val_df[0:50]#1769
    val_total = val_df.shape[0]
    test_df = pd.read_csv(test_file)
    test_total = test_df.shape[0]

    #log_file = "char_classifier_tweet_sentiment_log.txt"
    #----------------------------------------------------#
    # CHAR2IDX
    #----------------------------------------------------#
    char2idx_file = "tweet_50k_char2idx.json"
    #-----------------------------------------------------#
    #   lABEL DICT
    #-----------------------------------------------------#
    target_series = train_df.columns[1] #check before run
    label2idx,output_classes = label2index(train_df[target_series])
    #-----------------------------------------------------#
    #pytorch model and constant definination
    #-----------------------------------------------------#
    gpu = False
    if torch.cuda.is_available():
        gpu = True
        device_num = torch.cuda.device_count()
        print("cude available:")
        print(device_num)
    device = torch.device("cuda" if gpu else "cpu")

    text_classifier = classifier(output_classes,input_size=300,hidden_size=100)
    #text_classifier.load_state_dict(torch.load("best_classifier"))
    text_classifier = text_classifier.to(device)
    #-----------------------------------------------------#
    #pretrained_char_model,char2idx
    #-----------------------------------------------------#
    #char dicts
    with open(char2idx_file,'r') as f:
        char2idx = json.load(f)

    char_vocab_size = len(char2idx)

    char_model = char_lstm(char_vocab_size,char_embed=300,hidden = 150,char_layer = 1)
    #char_model.load_state_dict(torch.load("best_char_model_for_classifier"))
    char_model = char_model.to(device)
   #-------------------------------------------------------------#
   #               Training&evaluation
   #-------------------------------------------------------------#
    epoches = 15
    LR =0.1
    train_set = tweet_data(train_df)
    val_set = tweet_data(val_df)

    #os.mkdir("./char")
    log_file = "./char/setting.txt"
    with open(log_file,'w') as f:
        f.write("Char classifier from scatch, no pretraining\n")
        f.write("Num examples: %d \n train: %d, val: %d \n" %(output_classes,train_total,val_total))
        f.write("Start learning rate: %0.4f \n" %LR)

    train_loss,val_loss,train_accuracy,val_accuracy,best_val_loss,best_epoch=train(
                    train_set,train_total,val_set,val_total,
                    char2idx,label2idx,device,log_file,lr = LR,epoches = epoches)
    print("Training end")
    plt_train_val_loss(train_loss,val_loss,train_accuracy,val_accuracy)
    print("Best val loss %0.4f at epoch %d" %(best_val_loss,best_epoch))
