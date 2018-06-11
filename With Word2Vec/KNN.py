#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:47:58 2018

@author: isabelaruizroque
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:52:18 2018

@author: belar
"""
# To plot pretty figures
#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
#### ------------- IMPORTANDO BIBLIOTECAS ------------- ####
import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm

#tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from sklearn.preprocessing import scale

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer
import string
import unidecode

import simplejson

#### ------------- Parâmetros para a análise de sensibilidade paramétrica  ------------- ####
size_dimensions = 50
window_size = 5
m_count = 5
sg_algorithm = 1


#### ------------- Importando base de dados de emoções  ------------- ####

def ingest():

    data = pd.read_csv('/Users/isabelaruizroque/Documentos/Mackenzie - Macbook/Mestrado/Dissertação/Bases/wang_datasetComDisgeSurprise-2.txt', encoding= "ISO-8859-1")
    data['Emotion'] = list(data['Emotion'].map(str))
    data = data[data['Tweet'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print ('dataset loaded with shape', data.shape)
    return data

data = ingest()

indice = 0

# Transformando o texto com o nome das emoções em números {0, 1, 2, 3, 4, 5}
for word in data['Emotion']:
    
    if data['Emotion'][indice] == " happiness":
        data['Emotion'][indice] = 0
        
    elif data['Emotion'][indice] == " sadness":
        data['Emotion'][indice] = 1
    
    elif data['Emotion'][indice] == " anger":
        data['Emotion'][indice] = 2
        
    elif data['Emotion'][indice] == " fear":
        data['Emotion'][indice] = 3
        
    elif data['Emotion'][indice] == " disgust":
        data['Emotion'][indice] = 4
    
    elif data['Emotion'][indice] == " surprise":
        data['Emotion'][indice] = 5
        
    indice = indice + 1
    
#### ------------- Limpeza da base e Tokenization  ------------- ####

# Salvando as stopwords em inglês na variável stop
stop = stopwords.words('english')

# Símbolos que serão removidas em cada tweet
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

# Função para tokenization da base toda
def tokenize(tweet):
    try:
        # Transformando todo o tweet para lowercase
        tweet = str(tweet.lower())
        
        # Separando as palavras de cada tweet
        tokens = tokenizer.tokenize(tweet)
 
        # Removendo stopwords 
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        
        # Removendo símbolos
        tokens = [word for word in tokens if word not in punctuations]
        
        # Removendo aspas simples
        tokens = [word.replace("'", "") for word in tokens if "'" in punctuations]
        
        # Removendo acentos
        tokens = [unidecode.unidecode(word) for word in tokens]
        
        # Filtrando símbolos e removendo
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        tokens = list(filter(lambda t: not t.startswith('.'), tokens))
        tokens = list(filter(lambda t: not t.startswith(','), tokens))
        tokens = list(filter(lambda t: not t.startswith('!'), tokens))
        tokens = list(filter(lambda t: not t.startswith('?'), tokens))
        tokens = list(filter(lambda t: not t.startswith('"'), tokens))
        tokens = list(filter(lambda t: not t.startswith("'"), tokens))   
        tokens = list(filter(lambda t: not t.startswith(":"), tokens))
        tokens = list(filter(lambda t: not t.startswith(";"), tokens))
        tokens = list(filter(lambda t: not t.startswith(":"), tokens))
        tokens = list(filter(lambda t: not t.startswith("-"), tokens))
        tokens = list(filter(lambda t: not t.startswith("%"), tokens))
        tokens = list(filter(lambda t: not t.startswith("("), tokens))
        tokens = list(filter(lambda t: not t.startswith(")"), tokens))
        tokens = list(filter(lambda t: not t.startswith("="), tokens))
        tokens = list(filter(lambda t: not t.startswith("~"), tokens))
        tokens = list(filter(lambda t: not t.startswith("<"), tokens))
        tokens = list(filter(lambda t: not t.startswith("+"), tokens))
        tokens = list(filter(lambda t: not t.startswith(">"), tokens))
        tokens = list(filter(lambda t: not t.startswith("*"), tokens))
        tokens = list(filter(lambda t: not t.startswith("^"), tokens))
        tokens = list(filter(lambda t: not t.startswith("/"), tokens))
        tokens = list(filter(lambda t: not t.startswith("|"), tokens))
        tokens = list(filter(lambda t: not t.startswith("_"), tokens))
        
        # Realiza um join de todas as palavras, ao invés de retornar uma lista, é uma string
        return tokens
    except:
        return 'NC'

    
# Função de pós-processamento, realiza a tokenization e dropa tokens que contenham NC
def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['Tweet'].map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

# Chamando a função postprocess e salvando na variável data
data = postprocess(data)

#### ------------- Word2Vec  ------------- ####
tweet_w2v = Word2Vec(min_count=m_count, sg=sg_algorithm, window=window_size, size=size_dimensions)
tweet_w2v.build_vocab(x for x in tqdm(data['tokens']))

tweet_w2v.train([x for x in tqdm(data['tokens'])],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())]

# Salvando os word vectors em um arquivo
word_vectors_filename = "wordvectors_" + str(size_dimensions) + "_" + str(window_size) + str(sg_algorithm) + ".txt"
tweet_w2v.wv.save_word2vec_format(fname=word_vectors_filename, fvocab=None, binary=False)

# Quantidade de palavras no vocabulário
len(tweet_w2v.wv.vocab)

# Função para construir a entrada do classificador
def buildWordVector(tokens, size):
    
    # tokens = meus tokens = ['notices', 'almost', '300,000', 'mahomies', '33']
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) #* tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        print(count)
        #vec /= count
    return vec

# Construindo os vetores para dar entrada nos classificadores
train_vecs_w2v = np.concatenate([buildWordVector(z, size_dimensions) for z in tqdm(map(lambda x: x, data['tokens']))])
train_vecs_w2v = scale(train_vecs_w2v)

#### ------------- CLASSIFICAÇÃO  ------------- ####
from sklearn import cross_validation
from sklearn.svm import SVC

# Construindo o SVM
svm_class = SVC()

y = data['Emotion']
y=y.astype('int')

# Classificador SVM
from sklearn.model_selection import cross_val_score,StratifiedKFold

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

import itertools

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Positivo')
    plt.xlabel('Negativo')
    plt.savefig("MatrizConfusao-Exec_" + str(execucao)+ "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm))


import time
import datetime as dt

from sklearn import datasets
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
target_names = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
# shape of data is 150
#cv = StratifiedKFold(n_splits=10, shuffle=True)

# Definindo uma validação cruzada com 10 pastas
cv = StratifiedKFold(n_splits=10, shuffle=True)

from sklearn.model_selection import KFold

cvscores = []
mse_scores = []
loss_scores = []

acc_folder = []
f_score_folder = []

# Primeira execução
j = 1

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Contabilizando o tempo de início do classificador
start_time_execution_classificador = dt.datetime.now()
print('Início da execução do classificador {}'.format(str(start_time_execution_classificador)))

precision_folder = []

for execucao in range(1,11):
    
    # Contabilizando o tempo de início da execução
    start_time_execution = dt.datetime.now()
    
    #if (execucao == 0):
    #    print("EXECUÇÃO......................:", execucao+1)
    #else:
    #    print("EXECUÇÃO......................:",)
    
    print("EXECUÇÃO................................................:", execucao)
    
    print('Início da execução {}'.format(str(start_time_execution)))
    
    # Incrementando pasta
    j = 1
    
    aux_matrix = np.zeros((6,6))
    
    # Looping para as pastas
    for train_index, test_index in cv.split(train_vecs_w2v, y):
        
        # Contabilizando o tempo de início da pasta
        start_time_execution_folder = dt.datetime.now()
        print('Início da execução da pasta {}'.format(str(start_time_execution_folder)))

        # Incrementando as pastas
        if (j == 1):
            print("PASTA....................................: ", j)
        else:
            print("PASTA....................................: ", j)
            
        # Separando a base de dados
        X_tr, X_tes = train_vecs_w2v[train_index], train_vecs_w2v[test_index]
        y_tr, y_tes = y[train_index],y[test_index]
        
        # Fit do SVM
        print("FIT do KNN.................................")
        #clf = svm.SVC(kernel='linear', C=1).fit(X_tr, y_tr) 
        clf = knn.fit(X_tr, y_tr)
        # Predict do SVM
        print("PREDICT do KNN...")
        y_pred=clf.predict(X_tes)
        print("Predicting.................................")
        
        # Calculando acurácia
        print("Calculating Accuracy.................................")
        acc = accuracy_score(y_tes, y_pred)
        acc_folder.append(acc*100)
        print(acc)
    
        # Calculando precisão
        print("Calculating Precision................................")
        precision = precision_score(y_tes, y_pred, average='weighted')
        precision_folder.append(precision*100)
        print(precision)
       
        #Calculando Medida-F
        print("Calculating F-Score.................................")
        f_score = metrics.f1_score(y_pred=y_pred, y_true=y_tes, average='weighted')
        f_score_folder.append(f_score*100)
        print(f_score)        
 
        # Classification Report
        print("Classification Report...................")
        print(classification_report(y_tes, y_pred, target_names=target_names))
        class_report = classification_report(y_tes, y_pred, target_names=target_names)
        
        # Salvando o report de classificação
        f = open("KNN-ClassificationReport-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
        #simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(acc_folder), np.std(acc_folder))), f)
        simplejson.dump(class_report, f)
        f.close()

        # Incrementando pastas
        j = j + 1
        
        # Calculando a matriz de confusão para cada pasta
        cnf_matrix = confusion_matrix(y_tes, y_pred)
        np.set_printoptions(precision=2)
        
        # Salvar matriz de confusão em uma variável
        aux_matrix = aux_matrix + cnf_matrix
        
        # Transformando a matriz para o formato int64
        aux_matrix = aux_matrix.astype(np.int64)

    # Plotando a matriz de confusão
    plt.figure()
    plot_confusion_matrix(aux_matrix, classes=target_names,
                         title='Matriz de confusão')
    
    # Salvando a matriz de confusão das pastas para cada execução
    aux_matrix = aux_matrix.tolist()
    f = open("KNN-MatrizConfusao-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
    simplejson.dump(aux_matrix, f)
    f.close()
    
    #plt.show()
        
    # Salvando a acurácia das pastas para cada execução
    f = open("KNN-ACC-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
    simplejson.dump(acc_folder, f)
    f.close()
    
    # Salvando a acurácia média para cada execução
    f = open("KNN-Media-DesvioPadrao-ACC-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
    simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(acc_folder), np.std(acc_folder))), f)
    f.close()
    
    # Salvando a medida-F média para cada execução
    f = open("KNN-Media-DesvioPadrao-FSCORE-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
    simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(f_score_folder), np.std(f_score_folder))), f)
    f.close()

    # Salvando a Precisão média para cada execução
    f = open("KNN-Media-DesvioPadrao-PRECISION-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
    simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(precision_folder), np.std(precision_folder))), f)
    f.close()

    # Salvando o tempo de execução em um arquivo
    f = open("KNN-TempoDeExecucao-Exec_" + str(execucao) + "-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
    simplejson.dump('Inicio da execucao {}'.format(str(start_time_execution)), f)
    f.close()
    
    
    elapsed_time_execution = dt.datetime.now() - start_time_execution
    print('Tempo final da execução {}'.format(str(elapsed_time_execution)))
    

# Salvando a acurácia média para cada execução
f = open("KNN-ACURACIA-Media-DesvioPadrao-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(acc_folder), np.std(acc_folder))), f)
f.close()

# Salvando a medida-F média para cada execução
f = open("KNN-MEDIDAF-Media-DesvioPadrao-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(f_score_folder), np.std(f_score_folder))), f)
f.close()

# Salvando a precisão média para cada execução
f = open("KNN-PRECISION-Media-DesvioPadrao-Size_" + str(size_dimensions) + "-Window_" + str(window_size) + "-Sg_" + str(sg_algorithm) + ".txt", 'w')
simplejson.dump(("%.2f%% (+/- %.2f%%)" % (np.mean(precision_folder), np.std(precision_folder))), f)
f.close()

# Contabilizando o tempo de fim do classificador
end_time_execution_classificador = dt.datetime.now()

print('Fim da execução do classificador {}'.format(str(end_time_execution_classificador)))

#plt.savefig("MatrizConfusao-Exec_" + str(execucao))













