#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:37:44 2020

@author: yangmian
"""

import pandas as pd
import numpy as np
negative_review=pd.read_csv('../dataset/negative_review.txt',header=None)
positive_review=pd.read_csv('../dataset/positive_review.txt',header=None)

import re 

def getWords(feature,label,data):
    for i in range(len(data)):
        s=data.loc[i]
        if len(s[0])<800:
            f=re.findall(r'(\w+):(\d+)',s[0])
            feature[len(feature)]={F[0]:F[1] for F in f}
            label.append(re.findall(r'#label#:(\w+)',s[0])[0])
    return feature,label

feature,label=getWords({},[],negative_review)
feature,label=getWords(feature,label,positive_review)

def getWordRepresent(feature):
    word_set=set()
    for i in feature:
        for w in feature[i].keys():
            word_set.add(w)
    return word_set

word_set=getWordRepresent(feature)

def prepareData(feature,label,word_set):
    y=pd.DataFrame(data=label,columns=['label'])
    y[y['label']=='negative']=0
    y[y['label']=='positive']=1
    X=pd.DataFrame(data=np.zeros((len(feature),len(word_set))),columns=list(word_set))
    for i in feature:
        for word,freq in feature[i].items():
            X[word].loc[i]=int(freq)
    return X,y 

X,y=prepareData(feature,label,word_set)


