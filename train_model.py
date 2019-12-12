#!/usr/bin/env python
# coding: utf-8

# In[25]:


from model import Model
from data_processor import DataLoader
import json
import pickle
def main():
    config = json.load(open('config.json','r'))
    batch_size = config['batch_size']
    data_loader = DataLoader(config['data'],config['data_selector'],config['data_cols'])
    x,y = data_loader.get_train_batches(batch_size,config['seq_len'],config['vocab_dict_path'])
    if config['continue_training'] == True:
        model = Model.load_model(config['continue_training_info']['model_path'],config['vocab_dict_path'])
        model.train_model(x,y,batch_size,config['continue_training_info']['epochs'],config['continue_training_info']['save_dir'])
    elif config['new_training'] == False:
        model = Model(config['new_training_info']['embedding_out'],config['new_training_info']['rnn_units'],
                     config['batch_size'],config['vocab_dict_path'],config['new_training_info']['is_load'])
        model.train_model(x,y,batch_size,config['new_training_info']['epochs'],config['new_training_info']['save_dir'])

if __name__ == '__main__':
    main()


# In[ ]:




