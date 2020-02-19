from __future__ import print_function, division
import os
import sys
from nblearn import load_data
import random
import shutil

sample_fraction = 0.10

def sample_data(dataset_path):
    dataset = load_data(sys.argv[1].strip())
    spam_data = []
    ham_data = []
    for x, y in dataset:
        if(y == 'spam'):
            spam_data.append((x, y))
        if(y == 'ham'):
            ham_data.append((x, y))
            
    sample_size = (sample_fraction * len(dataset))
    random.shuffle(spam_data)
    random.shuffle(ham_data)
    
    return spam_data[:int(sample_size//2)] + ham_data[:int(sample_size//2)]
    

if __name__ == '__main__':
    
    sample_dataset = sample_data(sys.argv[1].strip())
    sample_data_dir = sys.argv[2].strip()
    
    for (x, y) in sample_dataset:
        os.makedirs(os.path.join(sample_data_dir, y), exist_ok=True)
        target_path = shutil.copy(x, os.path.join(sample_data_dir, y))
        
    