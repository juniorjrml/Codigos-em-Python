"""
as execuçoes desse arquivo fiz pelo google colab
devido ao baixo poder computacioinal

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

dataset = keras.datasets.fashion_mnist #Importando o dataset do modulo keras

(imagens_para_treino, train_labels), (test_images, test_labels) = dataset.load_data() #Separando os dados

nomes_das_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #Utilização dos indices de cada classe para sua representação

imagens_para_treino = imagens_para_treino / 255.0 #Normamlizando os valores de cada elemento(paradigma funcional -> para cada elemento de imagensparatreino dividir por 255)
test_images = test_images/255.0 #Idem

modelo_rede_neural = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #Primeira camada recebera (28,28) e sera transformada em uma entrada 28x28(de entrada)
    keras.layers.Dense(128,activation="relu"), #Segunda camada "totalmente conectada" com 128 nos
    keras.layers.Dense(10,activation="softmax") #Terceira camada(de saida) com 10 nos(correspondentes a cada classe)
    ])

modelo_rede_neural.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
modelo_rede_neural.fit(imagens_para_treino,train_labels,epochs=5)


perda_no_teste, acuracia_no_teste = modelo_rede_neural.evaluate(test_images,test_labels)
print("Perda no Teste:",perda_no_teste,"Acuracia no teste:",acuracia_no_teste)

predicao = modelo_rede_neural.predict(test_images)

"""
#Muito Tempo, muito exemplo para plotar
for i in enumerate(predicao):
  plt.grid(False)
  plt.imshow(test_images[i[0]],cmap=plt.cm.binary)
  plt.title("Classe:"+nomes_das_classes[test_labels[i[0]]])
  plt.xlabel("Predicao:"+nomes_das_classes[np.argmax(i[1])])
  plt.show()
"""

#reduzindo a quantidade de elementos para plotar
from random import *
quantidade_itens_para_mostrar = 5
numero_aleatorio = randint(quantidade_itens_para_mostrar, (len(test_images)-quantidade_itens_predicao) )#gerando um numero aleatrio que não ultrapasse o tamanho do vetor

imagens_teste_reduzido = test_images[numero_aleatorio:numero_aleatorio+quantidade_itens_predicao]#separando um trecho sequencial do teste
respostas_imagens_teste_reduzido = test_labels[numero_aleatorio:numero_aleatorio+quantidade_itens_predicao]#e separando o trecho correspondente nas respostas
predicao_reduzida = predicao[numero_aleatorio:numero_aleatorio+quantidade_itens_predicao]#e na predicao

#Uma vizualização amigavel sobre as predições feitas
for i in enumerate(predicao_reduzida):
  plt.grid(False)
  plt.imshow(imagens_teste_reduzido[i[0]],cmap=plt.cm.binary)
  plt.title("Classe:"+nomes_das_classes[respostas_imagens_teste_reduzido[i[0]]])
  plt.xlabel("Predicao:"+nomes_das_classes[np.argmax(i[1])])
  plt.show()
