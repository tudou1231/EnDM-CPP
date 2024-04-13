# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: train.py
# @Software: PyCharm

import os
import tensorflow as tf
import keras
from sklearn import metrics
from keras.backend import set_session
from keras.utils import np_utils

from sklearn.model_selection import KFold
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.disable_eager_execution()
from keras.callbacks import EarlyStopping

import numpy as np
np.random.seed(101)
from pathlib import Path


from model import base, BiGRU_base,lstm,MPMABP,transformer,gru,RNN,MLP,BiLSTM,TextLSTM,seq2seq,TextCNN,TextRNN,CNN_LSTM,CNN, Attention,MultiHeadAtten,Attention_BiLSTM,ANN,BiGRU_Attention,PositionalEmbedding_Transformer,transformer1



def train_my(train,test,test_last,  model_num, model_path,epoch,batch_si,hidden_size,jieduan):

    Path(model_path).mkdir(exist_ok=True)

    # data get
    if len(train)==2:
        X_train, y_train = train[0], train[1]

        y_train = keras.utils.to_categorical(y_train)
    else:
        X_train=train
    # train
    if X_train.ndim != 3 and model_path != 'PositionalEmbedding_Transformer':
        X_train = X_train.reshape(-1, 1, X_train.shape[1], )
    length = X_train.shape[1]
    if model_path != 'PositionalEmbedding_Transformer':
        length3 = X_train.shape[2]
    out_length = 2


    for counter in range(1, model_num+1):
        # get my neural network model
        if model_path == 'base':
            model = base(hidden_size, out_length, )
        elif model_path == 'BiGRU_base':
            model = BiGRU_base(length, length3, out_length, )
        elif model_path == 'lstm':
            model = lstm(hidden_size, length, length3, out_length, )
        elif model_path == 'TextLSTM':
            model = TextLSTM(hidden_size, out_length, )
        elif model_path == 'MPMABP':
            model = MPMABP(length, out_length, )
        elif model_path == 'transformer':
            model = transformer(length, out_length, )
        elif model_path == 'transformer1':
            model = transformer1(length, length3, out_length, )
        elif model_path == 'gru':
            model = gru(hidden_size, out_length, )
        elif model_path == 'RNN':
            model = RNN(hidden_size, out_length, )
        elif model_path == 'MLP':
            model = MLP(hidden_size, out_length, )
        elif model_path == 'TextCNN':
            model = TextCNN(length, length3, hidden_size, out_length, )
        elif model_path == 'seq2seq':
            model = seq2seq(length, length3, out_length, )
        elif model_path == 'BiLSTM':
            model = BiLSTM(hidden_size, out_length, )
        elif model_path == 'TextRNN':
            model = TextRNN(hidden_size, out_length, )
        elif model_path == 'CNN_LSTM':
            model = CNN_LSTM(hidden_size, out_length, )
        elif model_path == 'CNN':
            model = CNN(hidden_size, out_length, )
        elif model_path == 'ANN':
            model = ANN(length3, out_length, )
        elif model_path == 'Attention':
            model = Attention(length, length3, hidden_size, out_length, )
        elif model_path == 'MultiHeadAtten':
            model = MultiHeadAtten(length, length3, hidden_size, out_length, )
        elif model_path == 'Attention_BiLSTM':
            model = Attention_BiLSTM(length, length3, hidden_size, 32, out_length)
        elif model_path == 'BiGRU_Attention':
            model = BiGRU_Attention(length, length3, 32, out_length)
        elif model_path == 'PositionalEmbedding_Transformer':
            model = PositionalEmbedding_Transformer(length, 32, 6000, out_length)
        else:
            print('no model')


        if X_train.ndim!=3 and model_path!='PositionalEmbedding_Transformer' :
            X_train= X_train.reshape(-1, 1, length )
        if test[0].ndim != 3 and model_path != 'PositionalEmbedding_Transformer':
            length = test[0].shape[1]
            test[0] = test[0].reshape(-1, 1, length)
        es = EarlyStopping(patience=6)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        yan_train,yan_label=test_last
        yan_label= keras.utils.to_categorical(yan_label)

        if yan_train.ndim != 3 and model_path != 'PositionalEmbedding_Transformer':
            yan_train = yan_train.reshape(-1, 1,yan_train.shape[1], )
        model.fit(X_train, y_train, epochs=epoch, batch_size=batch_si, verbose=0,validation_data=(yan_train,yan_label ),
                  callbacks=[es])


        pred_res = model.predict(test[0])[:, 1]



        if jieduan == True :
            pred_train = model.predict(X_train)[:, 1]
            return pred_train, pred_res, model
        else:
            return pred_res, model

