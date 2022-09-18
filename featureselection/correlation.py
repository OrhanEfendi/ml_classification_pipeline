import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
class Remove_Correlated:
    def __init__(self,xtrain,xtest):
        self.xtrain=xtrain
        self.xtest=xtest
    def remove(self):
        # Create correlation matrix
        self.corr_matrix = self.xtrain.corr().abs()

        # Select upper triangle of correlation matrix
        self.upper =self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.85
        self.to_drop = [column for column in self.upper.columns if any(self.upper[column] > 0.85)]

        # Drop features
        self.xtrain=self.xtrain.drop(self.to_drop, axis=1)
        self.xtest=self.xtest[self.xtrain.columns]
        return self.xtrain,self.xtest