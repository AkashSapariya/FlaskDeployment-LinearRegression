# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:33:32 2022

@author: parth
"""

# Importing Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy import stats

dataset = pd.read_csv('D:/Data Science/youtube/krish naik/Deployment/Deployment FlaskAPI_LinearReg/Deployment-flask-master/hiring.csv')

dataset.head()

dataset['experience'].fillna(0, inplace = True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace = True)

X = dataset.iloc[:, :3]

# Converting Words To integer values:
    
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve':12,
                 'zero': 0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

# Splitting the dataset to training and testing

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model ith training data

regressor.fit(X,y)

# Saving model to disk

#pickle_dump(regressor, open('model.pkl', 'wb'))

# model = pickle.load(open('model.pkl', 'rb'))

pickle_out = open('regressor.pkl', 'wb')
pickle.dump(regressor, pickle_out)
pickle_out.close() 