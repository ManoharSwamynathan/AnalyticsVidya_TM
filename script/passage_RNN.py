#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:57:00 2017

@author: manohar
"""

import numpy as np
import pandas as pd
from lxml import html

from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.preprocessing import Tokenizer

 
import os
os.chdir('/home/manohar/Documents/AV_TM')

tr_data = pd.read_csv('train_MLWARE1.csv')

tr_data ['label'] = tr_data ['label'].replace(['sarcastic', 'non-sarcastic'], [1, 0])

trX = tr_data['tweet'].values
trY = tr_data['label'].values

print("Training data loaded and cleaned.")

tokenizer = Tokenizer(min_df=10, max_features=100000)
trX = tokenizer.fit_transform(trX)

print("Training data tokenized.")

layers = [
	Embedding(size=256, n_features=tokenizer.n_features),
	GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', seq_output=False, p_drop=0.75),
	Dense(size=1, activation='sigmoid', init='orthogonal')
]

model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
model.fit(trX, trY, n_epochs=6)

te_data = pd.read_csv('test_MLWARE1.csv')
ids = te_data['ID'].values
teX = te_data['tweet'].values
teX = tokenizer.transform(teX)
pr_teX = model.predict(teX).flatten()

pr_teX_cl = pr_teX  > 0.904 
pr_teX_cl = pr_teX_cl.astype(int)

sub = pd.DataFrame(np.asarray([ids, pr_teX_cl]).T)
sub.columns = ['ID', 'label'] 
sub['label'] = sub['label'].replace([1, 0],['sarcastic', 'non-sarcastic'])

sub.to_csv('submission.csv', index=False)



