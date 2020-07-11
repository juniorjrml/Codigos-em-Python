"""
Muito fraco, preciso explorar outros tutoriais sobre esse modelo e ver a melhor a preparação dos dados e formas de predição
"""
import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
digitos = load_digits()
dataset = scale(digitos.data)
y = digitos.target

k=10
samples,features = dataset.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

classificador = KMeans(n_clusters=k,init="random",n_init=10)
bench_k_means(classificador,"Classificador de Digitos",dataset)