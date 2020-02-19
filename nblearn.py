from __future__ import print_function, division
import os
import sys
import json
from collections import Counter

def load_data(path):
    if(not os.path.exists(path)):
        raise FileNotFoundError()
    
    dataset = []
    
    for root, dirs, files in os.walk(path):
        if('spam' in dirs or 'ham' in dirs):
            for d in ['spam', 'ham']:
                for file in os.listdir(os.path.join(root, d)):
                    if(file.split('.')[-1] == 'txt'):
                        dataset.append((os.path.join(root, d, file), d))
    return dataset


def create_count(path):
    c = Counter()
    with open(path, 'rt', encoding='latin1') as file:
        line = file.readline().strip()
        while(line):
            c.update(line.split())
            line = file.readline().strip()            
    return c

class Naive_bayes_model:
    def __init__(self):
        return
    
    def train(self, dataset):
        count_ham = 0
        count_spam = 0 
        count_token_spam = Counter()
        count_token_ham = Counter()
        
        for (x, y) in dataset:
            print(x, y)
            if(y == 'spam'):
                count_token_spam.update(create_count(x))
                count_spam += 1
            elif(y == 'ham'):
                count_token_ham.update(create_count(x))
                count_ham += 1
                
                
        vocabulary = set(list(count_token_ham.keys()) + list(count_token_spam.keys()))
        
        smoothing_flag = False
        for word in vocabulary:
            if(not word in count_token_ham):
                count_token_ham[word] = 1
                smoothing_flag = True
            if(not word in count_token_spam):
                count_token_spam[word] = 1
                smoothing_flag = True
                
        p_token_spam = {}
        p_token_ham = {}
        if(smoothing_flag):
            total_number_of_tokens_spam = sum([v for k, v in count_token_spam.items()]) + len(vocabulary)
            total_number_of_tokens_ham = sum([v for k, v in count_token_ham.items()]) + len(vocabulary)
        else:
            total_number_of_tokens_spam = sum([v for k, v in count_token_spam.items()])
            total_number_of_tokens_ham = sum([v for k, v in count_token_ham.items()])
        
        for k, v in count_token_spam.items():
            if(smoothing_flag):
                p_token_spam[k] = (count_token_spam[k]+1) / total_number_of_tokens_spam
            else:
                p_token_spam[k] = count_token_spam[k] / total_number_of_tokens_spam
            
            
        for k, v in count_token_ham.items():
            if(smoothing_flag):
                p_token_ham[k] = (count_token_ham[k]+1) / total_number_of_tokens_ham
            else:
                p_token_ham[k] = count_token_ham[k] / total_number_of_tokens_ham
                
            
        p_ham = count_ham / (count_ham+count_spam)
        p_spam = count_spam / (count_ham+count_spam)
        
        self.p_token_spam = p_token_spam  
        self.p_token_ham = p_token_ham
        self.p_spam = p_spam
        self.p_ham = p_ham
        
        return 
    
    
    def save(self, path):
        model_json = {'p_token_spam': self.p_token_spam, 
                      'p_token_ham': self.p_token_ham,
                      'p_spam': self.p_spam,
                      'p_ham': self.p_ham
                     }
        model_json = json.dumps(model_json)

        with open(path, 'wt') as file:
            file.write(model_json)
            
        return 
    
    def load(self, path):
        weights = None
        with open(path, 'rt') as file:
            weights = json.loads(file.read())
        self.p_token_spam = weights['p_token_spam']
        self.p_token_ham = weights['p_token_ham']
        self.p_spam = weights['p_spam']
        self.p_ham = weights['p_ham']
        
        return
    
    
    def predict(self, path_to_email):
        p_msg_spam = self.p_spam
        p_msg_ham = self.p_ham
        with open(path_to_email, 'r', encoding='latin1') as file:
            line = file.readline().strip()
            
            while(line):
                for word in line.split():
                    if(word in self.p_token_spam):
                        p_msg_spam *= self.p_token_spam[word]
                    
                    if(word in self.p_token_ham):
                        p_msg_ham *= self.p_token_ham[word]
                        
                line = file.readline().strip()
                
        return 'ham' if p_msg_ham >= p_msg_spam else 'spam'
                        

if __name__ == '__main__':
    data_path = sys.argv[1].strip()
    dataset = load_data(data_path)
    print(dataset[:10])
    model = Naive_bayes_model()
    model.train(dataset)
    print(model.p_ham, model.p_spam, model.p_token_spam)
    output_path = './nbmodel.txt'
    model.save(output_path)
