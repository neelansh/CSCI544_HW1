from __future__ import print_function, division
import os
import sys
import json
from collections import Counter
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nblearn import Naive_bayes_model, load_data
from nltk.stem import PorterStemmer

tokenizer = RegexpTokenizer(r'\w+')
stopwords_set = set(stopwords.words('english'))
alpha = 0.75

class NB_improved(Naive_bayes_model):
    def __init__(self):
        super().__init__()
    
    def create_count(self, path):
        c = Counter()
        with open(path, 'rt', encoding='latin1') as file:
            line = file.readline().strip()
            while(line):
                word_list = tokenizer.tokenize(line)
                word_list = [word for word in word_list if word not in stopwords_set]
                c.update(word_list)
                line = file.readline().strip()            
        return c

    def train(self, dataset):
        count_ham = 0
        count_spam = 0 
        count_token_spam = Counter()
        count_token_ham = Counter()
        
        for (x, y) in dataset:
            if(y == 'spam'):
                count_token_spam.update(self.create_count(x))
                count_spam += 1
            elif(y == 'ham'):
                count_token_ham.update(self.create_count(x))
                count_ham += 1
                
                
        vocabulary = set(list(count_token_ham.keys()) + list(count_token_spam.keys()))
        
        smoothing_flag = False
        for word in vocabulary:
            if(not word in count_token_ham):
                count_token_ham[word] = 0
                smoothing_flag = True
            if(not word in count_token_spam):
                count_token_spam[word] = 0
                smoothing_flag = True
                
        p_token_spam = {}
        p_token_ham = {}
        if(smoothing_flag):
            total_number_of_tokens_spam = sum([v for k, v in count_token_spam.items()]) + alpha*len(vocabulary)
            total_number_of_tokens_ham = sum([v for k, v in count_token_ham.items()]) + alpha*len(vocabulary)
        else:
            total_number_of_tokens_spam = sum([v for k, v in count_token_spam.items()])
            total_number_of_tokens_ham = sum([v for k, v in count_token_ham.items()])
        
        for k, v in count_token_spam.items():
            if(smoothing_flag):
                p_token_spam[k] = (count_token_spam[k]+alpha) / total_number_of_tokens_spam
            else:
                p_token_spam[k] = count_token_spam[k] / total_number_of_tokens_spam
            
            
        for k, v in count_token_ham.items():
            if(smoothing_flag):
                p_token_ham[k] = (count_token_ham[k]+alpha) / total_number_of_tokens_ham
            else:
                p_token_ham[k] = count_token_ham[k] / total_number_of_tokens_ham
                
            
        p_ham = count_ham / (count_ham+count_spam)
        p_spam = count_spam / (count_ham+count_spam)
        
        self.p_token_spam = p_token_spam  
        self.p_token_ham = p_token_ham
        self.p_spam = p_spam
        self.p_ham = p_ham
        
        return 
    
if __name__ == '__main__':
    data_path = sys.argv[1].strip()
    dataset = load_data(data_path)
    model = NB_improved()
    model.train(dataset)
    output_path = './nbmodel.txt'
    model.save(output_path)   
