import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string

import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
from tqdm import tqdm
import matplotlib.style as style
style.use('fivethirtyeight')
from sklearn.metrics import plot_roc_curve
from numpy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Layer

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFolD
from sklearn.metrics import confusion_matrix, classification_report


def read_data():
    # load data
    df = pd.read_csv('subset_documents.cvs', sep = '\t')
    # create binary label column based on date of death 
    df['death_outcome'] = np.where(df['Datum van overlijden'].isna() == True , 0, 1)
    df['death_outcome'] = pd.Categorical(df['death_outcome'])
    
    # fill NaN to empty string
    df = df.replace(np.nan, '', regex=True)   



def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('dutch'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers: word_tokenize('ebrahim^hazrati')'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

#remobe one character words
def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   

##remove specific word list
def remove_special_words(text):
    '''Remove the User predefine useless words from the text. The list should be in the lowercase.'''
    special_words_list=['af', 'iv', 'ivm', 'mg', 'dd', 'vrijdag','afspraak','over','met', 'van', 'patient', 'dr', 'geyik','heyman','bekker','dries','om', 'sel', 'stipdonk', 'eurling', 'knackstedt'
                        'lencer','volder','schalla']# list : words
    querywords=text.split()
    textwords = [word for word in querywords if word.lower() not in special_words_list]
    text=' '.join(textwords)
    return text
    
#%%
# Stemming with 'Snowball Dutch stemmer" package
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('dutch')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def lemma(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       
    return ' '.join(text_lemma)


#break sentences to individual word list
def sentence_word(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens
#break paragraphs to sentence token 
def paragraph_sentence(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token    


def tokenize(text):
    """Return a list of words in a text."""
    return re.findall(r'\w+', text)

def remove_numbers(text):
    no_nums = re.sub(r'\d+', '', text)
    return ''.join(no_nums)



def normalization_pitchdecks(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    remove_special_words,
    stem,
    remove_numbers
]
    for step in _steps:
        text=step(text)
    return text   



def clean_text(df):
    # apply all cleaning ufnctions on each of the 4 text columns
    # we don't join all text before cleaning, this allows to only train on certain text columns and disregarding others
    text_columns = ['Tekst1','Tekst2','Tekst3','Tekst4']
    
    for column in text_columns:
        df[column] = [normalization_pitchdecks(txt) for txt in df[column]]
        
        
    # create column of joined text
    df["joined_text"] = df["Tekst1"] +" "+ df["Tekst2"] +" "+ df['Tekst3'] +" "+ df['Tekst4']
    return df


def prepare_data(df):
    # Group by 'Patientnr' and 'label', and join together the different text fields for every patient
    combined = df.groupby(['Patnr', 'death_outcome'])['joined_text'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    
    # full text (per patient bc. of sequential data) to train model
    docs = combined['joined_text'].tolist()
    
    # label
    labels = combined['death_outcome'].values
    
    
    # check number of unique words to estimate a reasonable vocab size
    one_str = ''.join(docs)
    amount_unique_words = Counter(one_str.split())
    print(f'No. of unique words used in corpus: {amount_unique_words}') 
    
    return combined, docs, labels, amount_unique_words
    

def encode_documents(docs, word_amount):
    # translate words into one hot vectors according to corpus size of unique words
    # represent words as their position in the resulting sparse matrix
    # patient -> [435] instead of [0,0,0, ... 1,0,0,0] (1 is at pos. 435)
    encoded_docs = [one_hot(d, word_amount) for d in docs]
    return encoded_docs


def max_doc_length(encoded_docs):
	# find maximum number of words used in a single document
    return max([len(i) for i in encoded_docs])


def pad_docs(enc_docs, max_length):
	# pad all documents to standardize length - important for embedding layer
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs




def build_lstm(max_lenght, vocabe_size):
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = 32, input_length = max_length))
    model.add(Bidirectional(LSTM(128, activation='linear',return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation='linear')))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model



def create_partitions(padded_docs, labels):
    X = padded_docs
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def performance_check(model, y_test):
    vecint = np.vectorize(int)
    prediction=vecint((prediction_proba>0.519))

    print(confusion_matrix(y_test,prediction))
    print(classification_report(y_test,prediction))



if __name__ == '__main__':
	df = read_data()
	df_clean = clean_text(df) 
	combined_df, docs, labels, vocabe_size = prepare_data(df_clean)
	enc_docs = encode_documents(docs, vocabe_size)
	max_lenght = max_doc_length(enc_docs)
	padded_ = pad_docs(enc_docs, max_lenght)

	X_train, X_test, y_train, y_test = create_partitions(padded_, labels)

	model = build_lstm(max_lenght, vocabe_size)


	tensorboad = tf.keras.callbacks.TensorBoard(logdir=f'log/{datetime.datetime.now().strftime('%H-%M-%S')}')
	es = tf.keras.callbacks.EarlyStopping(monitor='val_loss' patience=5, mode='min')
	check = tf.keras.callbacks.ModelCheckpoint('model_checkpoint', monitor='val_loss')
	# fit the model
	model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_val, y_val), callbacks=[es, tensorboard, check])



	performance_check(model, y_test)