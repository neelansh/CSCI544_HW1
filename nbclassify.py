from __future__ import print_function, division
import os
import sys
import json
from nblearn import Naive_bayes_model, load_data


if __name__ == '__main__':
    model = Naive_bayes_model()
    model.load('./nbmodel.txt')
    print(model.p_spam)
    
    test_dataset = load_data(sys.argv[1].strip())
    
    with open('./nboutput.txt', 'wt') as file:
        for (x, y) in test_dataset:
            y_pred = model.predict(x)
            file.write("{} {}\n".format(y_pred, x))
            