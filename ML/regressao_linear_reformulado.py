import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import metrics
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import enum
#metrics.r2_score
#metrics.mean_poisson_deviance
class ModeloPredicaoRL:
    def __init__(self,modelo,metodoDeAvaliacao = metrics.r2_score):
        self.modelo = modelo
        self.tipo = ""
        self.metodoDeAvaliacao = metodoDeAvaliacao

    class MelhorTipoRetornoPredicao(enum.Enum):  # extend enum e impede alteração em tempo de execução dos valores
        """
        Classe criada para auxiliar na manutenção dos melhores tipos de retornos para predicao
        Tornando mais legivel o codigo
        """
        padrao = 0
        arredondado = 1
        truncado = 2

    def melhorPredicao(self,y_pred, y_verdadeiro):
        """
        :param y_pred: O que se quer avaliar como predicao
        :param y_verdadeiro: O resultado Real
        :return: Retorna a melhor forma de apresentação da predição
        """
        y_pred_arredondado = [round(p) for p in y_pred]
        y_pred_truncado = [int(p) for p in y_pred]

        acuracia_padrao = self.metodoDeAvaliacao(y_verdadeiro, y_pred)
        acuracia_arredondado = self.metodoDeAvaliacao(y_verdadeiro, y_pred_arredondado)
        acuracia_truncado = self.metodoDeAvaliacao(y_verdadeiro, y_pred_truncado)

        if (acuracia_truncado > acuracia_arredondado):
            return self.MelhorTipoRetornoPredicao.truncado
        elif (acuracia_arredondado > acuracia_padrao):
            return self.MelhorTipoRetornoPredicao.arredondado
        else:
            return self.MelhorTipoRetornoPredicao.padrao


    def fit(self,X,y):
        """
        :param X: dados para o treinamento e teste do modelo
        :param y: coluna alvo
        :return:
        """
        self.modelo.fit(X, y)  # usando a parte da base de dados separada para o o treino
        predicao = self.modelo.predict(X) #fazendo uma predição usando a parte da base de dados nao visto pelo modelo ainda
        self.tipo = self.melhorPredicao(predicao, y)

    def predict(self,X):
        """
        reescrito para retornar a predição no modo mais proveitoso para o modelo
        :param X: Dados a ser passados para a predição correspondente
        :return: lista com os elementos da predição
        """
        predicao = self.modelo.predict(X)
        if(self.tipo == self.MelhorTipoRetornoPredicao.padrao):
            return np.array(predicao)
        elif(self.tipo == self.MelhorTipoRetornoPredicao.arredondado):
            return np.array([round(p) for p in predicao])
        else:
            return np.array([int(p) for p in predicao])

    def score(self,X,y):
        X = np.array(X)
        y = np.array(y)
        return self.modelo.score(X,y)

    def salvar_modelo(self,nome_modelo = "regressao_linear"):
        with open(nome_modelo + ".pickle","wb") as f:  # salvando o modelo(sobrescreve o salvo anteriormente, na primeira execução cria o arquivo)
            pickle.dump(self, f)

    def carregar_modelo(nome_modelo = "regressao_linear"):
        carregamentoPickle = open(nome_modelo+".pickle", "rb")  # <-carregando objeto do arquivo
        return pickle.load(carregamentoPickle)  # <- carregando objeto do modelo previamente salvo

def acharModeloPorRepeticao(X, y, n_repeticoes=10, modelo = linear_model.LinearRegression(), nome_modelo = "regressao_linear"):
    #achando um melhor modelo nessas configurações(só e necessario executar na primeira vez)
    melhorPrecisao = 0 #flag para salvar o modelo de maior precisão
    for i in range(n_repeticoes):#tentando em n_repeticoes execuções para achar o melhor modelo possivel
        modeloRegressao = ModeloPredicaoRL(modelo)  #carregando um modelo sem nenhum parametro adicional
        x_treino, x_teste, y_treino, y_teste = sklearn.model_selection.train_test_split(X, y, test_size=0.3)  # separando a base para treinar o modelo e testar o mesmo
        modeloRegressao.fit(x_treino,y_treino) #passando a base para treinamento do modelo(a base e dividida em teste e treino)
        acuracia = modeloRegressao.score(X,y)#medindo a acuracia do modelo para a base inteira
        if acuracia>melhorPrecisao:
            melhorPrecisao = acuracia
            modeloRegressao.salvar_modelo()
    print(melhorPrecisao)




data = pd.read_csv("student/student-mat.csv", sep=";") #carregando dataset baixado de https://archive.ics.uci.edu
dados = ["G1", "G2", "G3", "studytime", "failures", "absences"] #dados que estamos julgando ser relevante para o modelo
data = data[dados] #separando da base de dados so os atributos(colunas,features) relevantes
alvo = "G3" #o que queremos "adivinhar"

y = np.array(data[alvo])
X = np.array(data.drop([alvo], 1))

acharModeloPorRepeticao(X,y)
modeloRegressaoLinear = ModeloPredicaoRL.carregar_modelo()
print(modeloRegressaoLinear.tipo)
print(modeloRegressaoLinear.tipo)
predicao = modeloRegressaoLinear.predict(X)
acuracia2 = modeloRegressaoLinear.score(y,predicao)

for x in enumerate(predicao):# iterando sobre a predição para ter uma vizualização dos "chutes" feitos e quao perto foi
    #utilizei o enumerate para um controlhe melhor e mais bonito do indice do que em range(len(predicao))
    print("Predição: {:.2f}".format(x[1]), "Valor Real:", y[x[0]])

"""
for i in dados: #lista com os nomes das features relevantes
    labelEixoX = i
    style.use("ggplot")
    pyplot.scatter(data[labelEixoX],data[alvo])
    pyplot.xlabel(labelEixoX)
    pyplot.ylabel(alvo)
    pyplot.show()
"""