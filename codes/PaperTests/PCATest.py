# -*- coding: utf-8 -*-

"""
@author: Mark Wu
@file: PCATest.py
@time: 2023/3/25 10:26
"""

from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd



class PrincipalComponentAnalysis():
    def __init__(self, nData, nft):        # nData表示原始数据，nft表示指定特征数
        self.data = nData
        self.nFeatures = nft
        self.pca = PCA()
        self.bestFeatures = 0
    def getBestFeatures(self):
        return self.bestFeatures

    def compute(self):
        # First center and scale the data
        scaled_data = preprocessing.scale(self.data)
        #Ajust the model and execute
        self.pca.fit(scaled_data)
        #pca_data = self.pca.transform(scaled_data)

        #Create a bar graph of PC's and yours variance
        '''
        per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
        plt.bar(x=range(1,len(per_var)+1),height=per_var, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal component')
        plt.title('Scree Plot')
        plt.show()'''

        # Determine which the biggest influence on PC1
        # loading_scores = pd.Series(self.pca.components_[0], index=ft)   # ft为特征名列表
        loading_scores = pd.Series(self.pca.components_[0])
        ## now sort the loading scores based on their magnitude
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        # get the names of the top X features
        self.bestFeatures = sorted_loading_scores[0:self.nFeatures].index.values
        # print the names and their scores (and +/- sign)
        print(loading_scores[self.bestFeatures])
