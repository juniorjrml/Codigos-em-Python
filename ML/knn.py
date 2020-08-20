import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model, preprocessing
"""
modelo interessante para conjunto menores devido ao alto custo computacional
apresenta alguns problemas que deve ser analisado a cada dataset com o parametro K:
se grande ou pequeno demais pode gerar classificar errada, pois pode "pegar os vizinhos errados"
computacionalmente falando não e interessante salvar o modelo
"""
data =  pd.read_csv("Car-Data-Set/car.data")
preprocessador = preprocessing.LabelEncoder()

'''
apresentação dos dados:

buying: vhigh, high, med, low.
maint: vhigh, high, med, low.
doors: 2, 3, 4, 5more.
persons: 2, 4, more.
lug_boot: small, med, big.
safety: low, med, high. 

'''
"""
converte classes não numericas em numeros(como um id para a classe
"""
buying = preprocessador.fit_transform(list(data["buying"]))
maint = preprocessador.fit_transform(list(data["maint"]))
door = preprocessador.fit_transform(list(data["door"]))
persons = preprocessador.fit_transform(list(data["persons"]))
lug_boot = preprocessador.fit_transform(list(data["lug_boot"]))
safety = preprocessador.fit_transform(list(data["safety"]))
class_ = preprocessador.fit_transform(list(data["class_"]))
#para um nome mais compativel troquei o nome do atributo class para class_ pois class é palavra reservada em python

X = list(zip(buying,maint,door,persons,lug_boot,safety))
"""
pega um conjunto de listas e transforma em um obj(por isso o cast de lista)
que representa uma lista onde cada elemento i é uma tupla de todos os iesimos
elementos de cada lista ex.:
a = [1,2,3]
b = [4,5,6]
c= list(zip(a,b))
c é igual a [(1,4),(2,5),(3,6)]
"""
y =list(class_)

x_treino, x_teste, y_treino, y_teste = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
modeloClassificacao = KNeighborsClassifier(n_neighbors=5)
modeloClassificacao.fit(x_treino,y_treino)
acuracia = modeloClassificacao.score(x_teste,y_teste)
predicao = modeloClassificacao.predict(x_teste)
print(acuracia)