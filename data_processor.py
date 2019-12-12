#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import re
import pickle

class DataLoader():
    
    def __init__(self,file_name,selector,value_cols):
        df = pd.read_csv(file_name)
        for key in selector.keys():
            if key is None:
                break
            df = df[df[key]==selector[key]]
        for col in value_cols:
            df[col] = df[col].str.lower()
            df[col] = df[col].str.replace(r'http[\w:/\.]+','') # remove urls
            df[col] = df[col].str.replace(r'[^!\'"#$%&\()*+,-./:;<=>?@_’`{|}~\w\s]',' ') #remove everything but characters and punctuation
            df[col] = df[col].str.replace(r'b[\d\w\W\D]+@','@')
            df[col] = df[col].str.replace(r'x[\d\w]{2}','')
            df[col] = df[col].str.replace(r'\\*','')
            df[col] = df[col].str.replace(r'\\\'','\'')
            df[col] = df[col].str.replace(r'&amp','')
            df[col] = df[col].str.replace(r'\s\s+',' ') #replace multple white space with a single one
            df = df[[len(str(t))<180 for t in df[col].values]]
            df = df[[len(str(t))>50 for t in df[col].values]]
        self.data = [text for text in df.get(value_cols).get_values()[::-1].flatten()]
       
     #获取每个word的查找表及转换表 './tweets.pickle'
    def _save_vocabulary_dict(self,filepath):
        all_sentences = ''.join(self.data)
        chars = set(all_sentences)
        sorted_chars = sorted(chars)
        char2int = dict(zip(sorted_chars,range(len(sorted_chars))))
        char2int['<Go>'] = len(char2int)
        char2int['<End>'] = len(char2int)
        char2int['<Pad>'] = len(char2int)
        int2char = dict(zip(char2int.values(),char2int.keys()))
        with open(filepath, 'wb') as f:
            pickle.dump((char2int, int2char), f)
        return char2int,int2char
    
    #为了使用stateful lstm，必须将x,y分成每个batch都含有相同数量sample的batches
    def get_train_batches(self,batch_size, seq_length,vocab_path):
        """
        Return batches of input and target
        :param int_text: Text with the words replaced by their ids
        :param batch_size: The size of batch
        :param seq_length: The length of sequence
        :return: Batches as a Numpy array
        """
        char2int,int2char = self._save_vocabulary_dict(vocab_path)
        self.text_num = [[char2int['<Go>']]+ [char2int[c] for c in sentence] + [char2int['<End>']] for sentence in self.data ]
        int_text = []
        for t in self.text_num:
            int_text+=t
        slice_size = batch_size * seq_length
        n_batches = len(int_text) // slice_size
        x = int_text[: n_batches*slice_size]
        y = int_text[1: n_batches*slice_size + 1]

        x = np.split(np.reshape(x,(batch_size,-1)),n_batches,1)
        y = np.split(np.reshape(y,(batch_size,-1)),n_batches,1)
        x = np.vstack(x)
        y = np.vstack(y)
        y = y.reshape(y.shape+(1,))
        return x, y


# In[ ]:




