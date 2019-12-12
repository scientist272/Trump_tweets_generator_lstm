#!/usr/bin/env python
# coding: utf-8

# In[25]:


from model import Model
from data_processor import DataLoader
import json
import pickle
def main():
    config = json.load(open('config.json','r'))
    if config['use_exist_model'] == True:
        model = Model.load_model(config['use_exist_model_path'],config['vocab_dict_path'])
    else:
        print('modify your config.json first, set use_exist_model = true ')
        exit()
    
    while(True):
        input_sentence = input('Input some sentences, if you input -random, it will generate a random sentence. Input -exit to quit the program \n')
        if input_sentence == '-random':
            
            print('\nTrump would say: '+model.generate_random_sentence(150)+'\n')
        elif input_sentence== '-exit':
            break
        else:    
            print('\nTrump would say: '+model.generate_sentence_with_start(150,input_sentence)+'\n')

if __name__ == '__main__':
    main()


# In[ ]:
