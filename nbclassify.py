from __future__ import print_function, division
import os
import sys
import json
from nblearn import Naive_bayes_model


if __name__ == '__main__':
    model = Naive_bayes_model()
    model.load('./nbmodel.txt')
    print(model.p_spam)
    pass