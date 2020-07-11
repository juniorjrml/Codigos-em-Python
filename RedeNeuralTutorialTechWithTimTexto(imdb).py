"""
as execuçoes desse arquivo fiz pelo google colab
devido ao baixo poder computacioinal

"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

dataset = keras.datasets.imdb
index_das_palavras = dataset.get_word_index() #carrega os indices e itens do modelo
index_das_palavras = {k:(v+4) for k,v in index_das_palavras.items()} #Cria um dicionario com as palvras e seus indices chaves começando do indice 4

index_das_palavras["<PAD>"] = 0 #Usado para padronizar
index_das_palavras["<START>"] = 1 #
index_das_palavras["<UNK>"] = 2  # unknown
index_das_palavras["<UNUSED>"] = 3 #
palavras_dos_indexs = dict([(palavra,chave) for (chave,palavra) in index_das_palavras.items()]) #criando um dicionario  que usa como chave as palavras e os indices como valores

tamanho_padrao_de_entrada = 250 #tamanho para padronizar o que sera (tamanho do input) inserido na rede
(dados_para_treino, respostas_do_treino), (dados_para_teste, respostas_dos_testes) = dataset.load_data(num_words=10000)
dados_para_treino = keras.preprocessing.sequence.pad_sequences(dados_para_treino,value=index_das_palavras["<PAD>"],padding="post",maxlen=tamanho_padrao_de_entrada) #Padronizando tamanho dos dados
dados_para_teste = keras.preprocessing.sequence.pad_sequences(dados_para_teste,value=index_das_palavras["<PAD>"],padding="post",maxlen=tamanho_padrao_de_entrada) #Padronizando tamanho dos dados


def decode_review(text):
    return "".join([palavras_dos_indexs.get(i, "?") for i in text])

rede = keras.Sequential()

rede.add(keras.layers.Embedding(100000,16)) #Primeira camada ajuda a achar semelhanças entre palavras(ou frases?)/(pelo o que eu entendi)
rede.add(keras.layers.GlobalAveragePooling1D()) #Segunda camada reduz o numero de dimenções para facilitar o processo
rede.add(keras.layers.Dense(16,activation="relu")) #camada cheia(full conected)
rede.add(keras.layers.Dense(1,activation="sigmoid")) #Camada de saida

rede.summary()
rede.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

x_validacao = dados_para_treino[:10000]
x_treino = dados_para_treino[10000:]

y_validacao = respostas_do_treino[:10000]
y_treino = respostas_do_treino[10000:]

treino_do_modelo = rede.fit(x_treino,y_treino,epochs=40,batch_size=512,validation_data=(x_validacao,y_validacao),verbose = 1)
resultados = rede.evaluate(dados_para_teste,respostas_dos_testes)
print(resultados)