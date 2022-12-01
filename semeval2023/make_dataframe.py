import pandas as pd
from tqdm import tqdm
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import CountVectorizer
import argparse



def make_dataframe(input_folder, labels_folder=None):
    #MAKE TXT DATAFRAME
    text = []
    
    for fil in tqdm(filter(lambda x: x.endswith('.txt'), os.listdir(input_folder))):

        iD, txt = fil[7:].split('.')[0], open(input_folder +fil, 'r', encoding='utf-8').read() 
        text.append((iD, txt))

    df_text = pd.DataFrame(text, columns=['id','text']).set_index('id')

    df = df_text

    #MAKE LABEL DATAFRAME
    if labels_folder:
        labels = pd.read_csv(labels_folder, sep='\t', header=None)
        labels = labels.rename(columns={0:'id',1:'type'})
        labels.id = labels.id.apply(str)
        labels = labels.set_index('id')

        #JOIN
        df = labels.join(df_text)[['text','type']]

    return df

def main():

    folder_train = "semeval2023/data_en_subtask1/train-articles-subtask-1"
    folder_dev = "semeval2023/data_en_subtask1/dev-articles-subtask-1"
    labels_train_fn = "semeval2023/data_en_subtask1/train-labels-subtask-1.txt"

    #Read Data
    print('Loading training...')
    train = make_dataframe(folder_train, labels_train_fn)
    print('Loading dev...')
    test = make_dataframe(folder_dev)

    X_train = train['text'].values
    X_test = test['text'].values
    Y_train = train['type'].values