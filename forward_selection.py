# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:17:16 2020

@author: desdina.kof
"""

import pandas as pd
import statsmodels.api as sm
from pandas import read_excel
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import xlrd


def forward_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    initial_list =[]
    included = list(initial_list)
    while True:
        changed = False
        #forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index = excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval <threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  with p-value'.format(best_feature, best_pval))
        
        if not changed:
            break
        
    return included

file_errors_location = "C:/Users/desdina.kof/Desktop/ulak/ulak/TURKCELL_sile.csv"
df = pd.read_excel(file_errors_location)

y = df['RRC Setup Success Rate Bin'] # drops the column
x = df.drop('RRC Setup Success Rate Bin', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

forward_selection(X_train, y_train)