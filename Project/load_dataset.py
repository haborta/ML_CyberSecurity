# *****************
# Used to load and pre-processing fake news datasets
# *****************

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

class kaggle_dataset:
    def __init__(self, path="./dataset/Fake-News/"):
        self.path = path
        
        df_fake_news = pd.read_csv(self.path+'Fake.csv')
        df_true_news = pd.read_csv(self.path+'True.csv')
        df_fake_news['fake'] = 1
        df_true_news['fake'] = 0
        # Concatenate fake and true news
        df_news = pd.concat([df_fake_news, df_true_news])
        # Dataframe shuffling and feature and label extraction
        df_news = df_news.sample(frac=1) # frac=1 means to return all rows (in random order)

        self.train_sentences, self.val_sentences, self.train_labels, self.val_labels=train_test_split(
            df_news['text'].to_numpy(),
            df_news['fake'].to_numpy(),
            test_size=0.2,
            random_state=42)

class kaggle_dataset_title:
    def __init__(self, path="./dataset/Fake-News/"):
        self.path = path
        
        df_fake_news = pd.read_csv(self.path+'Fake.csv')
        df_true_news = pd.read_csv(self.path+'True.csv')
        df_fake_news['fake'] = 1
        df_true_news['fake'] = 0
        # Concatenate fake and true news
        df_news = pd.concat([df_fake_news, df_true_news])
        # Dataframe shuffling and feature and label extraction
        df_news = df_news.sample(frac=1) # frac=1 means to return all rows (in random order)

        self.train_sentences, self.val_sentences, self.train_labels, self.val_labels=train_test_split(
            df_news['title'].to_numpy(),
            df_news['fake'].to_numpy(),
            test_size=0.2,
            random_state=42)
        
class liar_dataset:
    def __init__(self, path="./dataset/Liar/"):
        self.path = path
        
        def to_label(a):
            a_cat = [0]*len(a)
            for i in range(len(a)):
                if a[i]=='true':
                    a_cat[i] = 1
                elif a[i]=='mostly-true':
                    a_cat[i] = 1
                elif a[i]=='half-true':
                    a_cat[i] = 1
                elif a[i]=='barely-true':
                    a_cat[i] = 0
                elif a[i]=='false':
                    a_cat[i] = 0
                elif a[i]=='pants-fire':
                    a_cat[i] = 0
                else:
                    print('Incorrect label')
            return a_cat
        
        df_liar_train = pd.read_csv(self.path+'train.tsv', sep="\t", header=None)
        df_liar_test = pd.read_csv(self.path+'test.tsv', sep="\t", header=None)

        # Fill nan (empty boxes) with 0
        df_liar_train = df_liar_train.fillna(0)
        df_liar_test = df_liar_test.fillna(0)

        self.train_sentences = df_liar_train[2].to_numpy()
        self.val_sentences = df_liar_test[2].to_numpy()
        self.train_labels = np.array(to_label(df_liar_train[1]))
        self.val_labels = np.array(to_label(df_liar_test[1]))
