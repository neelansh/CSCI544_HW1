from __future__ import print_function, division
import os
import sys
import json

def get_gt(path):
    if('spam' in path.split('.')):
        return 'spam'
    else:
        return 'ham'
    
    
def calculate_precision_recall_f1(correctly_classified_c, classified_c, belongs_c):
    precision = correctly_classified_c / classified_c if classified_c != 0 else 0
    recall = correctly_classified_c / belongs_c if belongs_c != 0 else 0
    
    f1 = (2*precision*recall) / (precision+recall) if (precision != 0 and recall != 0) else 0
    
    return (precision, recall, f1)
    

if __name__ == '__main__':
    correctly_classified_spam = 0
    correctly_classified_ham = 0
    classified_spam = 0
    classified_ham = 0
    belongs_in_spam = 0
    belongs_in_ham = 0
    
    with open('./nboutput.txt', 'rt') as file:
        line = file.readline().strip()
        
        while(line):
            y_pred, y_gt = line.split('\t')
            y_gt = get_gt(y_gt)
            
            belongs_in_spam += 1 if y_gt == 'spam' else 0
            belongs_in_ham += 1 if y_gt == 'ham' else 0
            
            classified_spam += 1 if y_pred == 'spam' else 0
            classified_ham += 1 if y_pred == 'ham' else 0
            
            if(y_pred == y_gt):
                if(y_gt == 'spam'):
                    correctly_classified_spam += 1
                elif(y_gt == 'ham'):
                    correctly_classified_ham += 1
            
            line = file.readline().strip()
            
    precision_spam, recall_spam, f1_spam = calculate_precision_recall_f1(correctly_classified_spam, classified_spam, belongs_in_spam)
    
    precision_ham, recall_ham, f1_ham = calculate_precision_recall_f1(correctly_classified_ham, classified_ham, belongs_in_ham)
    
    
    print("Precision Spam", precision_spam)
    print("Precision Ham", precision_ham)
    print("Recall Spam", recall_spam)
    print("Recall Ham", recall_ham)
    print("F1 Spam", f1_spam)
    print("F1 Ham", f1_ham)
