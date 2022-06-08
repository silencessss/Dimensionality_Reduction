"""
Created on Thu Jun 08 04:04:29 2022
@author: PeterChan
#reference: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
"""
from __future__ import print_function
from operator import ge
import time
import numpy as np
import pandas as pd
import keras
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#import DataLoader

class config:
    TARGET_H = 28
    TARGET_W = 28
    CLASSES = 10
class load_data:
    # ::output:: (data value, pixel size), (label value)
    # ::example:: (70000, 784) (70000,)
    def get_mnist():
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32')/255.0
        x_train = x_train.reshape(60000,784)
        return x_train,y_train
    def get_iris():
        iris = load_iris()
        x = iris.data
        y = iris.target
        return x,y
    def get_AcneECK():
        train_x,train_y,test_x,test_y,labelsarr = DataLoader.AcneECK.byDir()
        train_x = train_x.astype('float32')/255.0
        train_x = train_x.reshape(460,50176)
        train_y = np.array(train_y)
        print('labelsarr',labelsarr.shape)
        
        return train_x,labelsarr
class visualization:
    def input_data(feat_cols):
        #plt.gray()
        fig = plt.figure( figsize=(16,15) )
        for i in range(0,15):
            ax = fig.add_subplot(3,5,i+1, title="Label: {}".format(str(df.loc[rndperm[i],'label'])) )
            ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((config.TARGET_W,config.TARGET_H)).astype(float))
        plt.show()

def transform(X,y):
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    print('Size of the dataframe: {}'.format(df.shape))
    return df,feat_cols

def randomly(df):
    return np.random.permutation(df.shape[0])

class dimensionality:
    def tSNE_dimensionality(df,feat_cols,rndperm):
        N = 10000
        df_subset = df.loc[rndperm[:N],:].copy()
        data_subset = df_subset[feat_cols].values
        
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        plt.figure()#figsize=(16,10)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", config.CLASSES),
            data=df_subset,
            legend="full",
            alpha=0.3
        ).set(title='T-SNE projection')
        plt.show()

    def PCA_dimensionality(df,feat_cols,rndperm):
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[feat_cols].values)
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        df['pca-three'] = pca_result[:,2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        # 二維可視
        plt.figure()
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", config.CLASSES),#color number = classes
            data=df.loc[rndperm,:],
            legend="full",
            alpha=0.3
        ).set(title='PCA-2d projection')

        # 三維可視
        ax = plt.figure().gca(projection='3d')
        ax.scatter(
            xs=df.loc[rndperm,:]["pca-one"], 
            ys=df.loc[rndperm,:]["pca-two"], 
            zs=df.loc[rndperm,:]["pca-three"], 
            #c=df.loc[rndperm,:]["y"], 
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.show()




X,y = load_data.get_mnist()
df,feat_cols = transform(X,y)
rndperm = randomly(df)
#visualization.input_data(feat_cols)
dimensionality.PCA_dimensionality(df,feat_cols,rndperm)
dimensionality.tSNE_dimensionality(df,feat_cols,rndperm)