#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, TimeDistributed
from keras.models import load_model, model_from_json,save_model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import pickle
import os
import datetime as dt

class Model():
    
    def __init__(self,embedding_out,rnn_units,batch_size,vocab_dic_path,is_load):
        # Load text Dict
        with open(vocab_dic_path, 'rb') as f:
            char2int, int2char = pickle.load(f)
            self.char2int = char2int
            self.int2char = int2char
        len_vocab = len(char2int)
        if is_load == False:
            self.stateful_model = Model._build_stateful_model(embedding_out,rnn_units,len_vocab,batch_size)
        self.predict_model = Model._build_predict_model(embedding_out,rnn_units,len_vocab)

    @classmethod
    def _build_stateful_model(cls,embedding_out,rnn_units,len_vocab,batch_size):
        model = Sequential()
        model.add(Embedding(len_vocab, embedding_out, batch_size=batch_size)) 
        model.add(LSTM(rnn_units, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
        model.add(LSTM(rnn_units, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(Dense(len_vocab, activation='softmax')))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        print('='*100)
        model.summary()
        print('='*100)
        return model
    @classmethod
    def _build_predict_model(cls,embedding_out,rnn_units,len_vocab):
        predict_model = Sequential()
        predict_model.add(Embedding(len_vocab, embedding_out)) 
        predict_model.add(LSTM(rnn_units, return_sequences=True))
        predict_model.add(Dropout(0.2))
        predict_model.add(LSTM(rnn_units, return_sequences=True))
        predict_model.add(Dropout(0.2))
        predict_model.add(TimeDistributed(Dense(len_vocab, activation='softmax')))
        print('='*100)
        predict_model.summary()
        print('='*100)
        return predict_model
    
    def generate_sentence_with_start(self,sentence_len,start_sentence):
        letter = [self.char2int[c] for c in start_sentence]
        sentence = [self.int2char[l] for l in letter]
        self.predict_model.set_weights(self.stateful_model.get_weights())
        for i in range(150):
            if self.int2char[letter[-1]] == '<End>':
                break
            p = self.predict_model.predict(np.array(letter)[None,:])
            letter.append(np.random.choice(len(self.char2int),1,p=p[0][-1])[0])
            if self.int2char[letter[-1]]!='<End>':
                sentence.append(self.int2char[letter[-1]])
        return ''.join(sentence)
    
    def generate_random_sentence(self,sentence_len):
        sentence = []
        letter = [self.char2int['<Go>']] #choose a random letter
        self.predict_model.set_weights(self.stateful_model.get_weights())
        for i in range(sentence_len):
            if self.int2char[letter[-1]] == '<End>':
                sentence.append(self.int2char[letter[-1]])  
                break
            sentence.append(self.int2char[letter[-1]])
            p = self.predict_model.predict(np.array(letter)[None,:])
            letter.append(np.random.choice(len(self.char2int),1,p=p[0][-1])[0])
        return ''.join(sentence)

    def train_model(self,x,y,batch_size,epochs,save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)]
        for i in range(0,epochs):
            self.stateful_model.fit(x,y, batch_size=batch_size, epochs=1, shuffle=False,verbose=1,callbacks=callbacks)
            self.stateful_model.reset_states()
    
    @classmethod
    def load_model(cls,filename,vocab_path):
        temp = load_model(filename)
        embedding_out = temp.get_config()['layers'][0]['config']['output_dim']
        rnn_units = temp.get_config()['layers'][1]['config']['units']
        batch_size = temp.get_config()['layers'][0]['config']['batch_input_shape'][0]
        result = Model(embedding_out,rnn_units,batch_size,vocab_path,True)
        result.stateful_model = temp
        return result
