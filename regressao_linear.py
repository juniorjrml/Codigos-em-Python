import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import metrics
"""
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
"""
import pickle

def procurar_modelo_por_repeticao(X,y,repeticoes = 100,tamanho_teste = 0.3):
    #achando um melhor modelo nessas configurações(só e necessario executar na primeira vez)
    try:
        carregamentoPickle = open("regressao_linear.pickle", "rb")#<-carregando objeto do arquivo
        modeloRegressaoLinear = pickle.load(carregamentoPickle)   #<-carregando objeto do modelo previamente salvo
        x_treino, x_teste, y_treino, y_teste = sklearn.model_selection.train_test_split(X, y,test_size=tamanho_teste)  # separando a base para treinar o modelo e testar o mesmo
        melhorPrecisao = modeloRegressaoLinear.score(x_teste, y_teste)  # testando o modelo com dados não visto por ele ainda
    except:
        melhorPrecisao = 0 #flag para salvar o modelo de maior precisão
    for i in range(repeticoes):#tentando em 100 execuções achar o melhor modelo possivel
        x_treino, x_teste, y_treino, y_teste = sklearn.model_selection.train_test_split(X, y, test_size=0.3)#separando a base para treinar o modelo e testar o mesmo
        modeloRegressaoLinear = linear_model.LinearRegression()#carregando o modelo de regressao linear sem nenhum parametro adicional
        modeloRegressaoLinear.fit(x_treino, y_treino)#usando a parte da base de dados separada para o o treino
        predicao = modeloRegressaoLinear.predict(x_teste)
        acuracia = modeloRegressaoLinear.score(x_teste, y_teste)#testando o modelo com dados não visto por ele ainda
        if acuracia>melhorPrecisao:
            melhorPrecisao = acuracia
            with open("regressao_linear.pickle", "wb") as f:# salvando o modelo caso seja a melhor acuracia(sobrescreve o salvo anteriormente, na primeira execução cria o arquivo)
                pickle.dump(modeloRegressaoLinear, f)

data = pd.read_csv("student/student-mat.csv", sep=";")#carregando dataset baixado de https://archive.ics.uci.edu
dados = ["G1", "G2", "G3", "studytime", "failures", "absences"] #dados que estamos julgando ser relevante para o modelo
data = data[dados] #separando da base de dados so os atributos(colunas,features) relevantes
alvo = "G3" #o que queremos "adivinhar"

y = np.array(data[alvo])
X = np.array(data.drop([alvo], 1))

procurar_modelo_por_repeticao(X,y,repeticoes=100,tamanho_teste=0.5)
carregamentoPickle = open("regressao_linear.pickle", "rb")#<-carregando objeto do arquivo
modeloRegressaoLinear = pickle.load(carregamentoPickle)   #<-carregando objeto do modelo previamente salvo


predicao = modeloRegressaoLinear.predict(X)
predicao = [abs(i) for i in predicao]
metodos_de_score = [metrics.mean_absolute_error,metrics.mean_poisson_deviance,metrics.explained_variance_score,metrics.r2_score,metrics.max_error]
print("Metodo de Avaliação:", modeloRegressaoLinear.score.__name__, "Pontuação:", "{:.2f}".format(modeloRegressaoLinear.score(X,y)))
for metodo in metodos_de_score:
    acuracia = metodo(y,predicao)
    print("Metodo de Avaliação:", metodo.__name__,"Pontuação:","{:.2f}".format(acuracia))

"""
###visualizar graficamente:
for i in dados: #lista com os nomes das features relevantes
    labelEixoX = i
    style.use("ggplot")
    pyplot.scatter(data[labelEixoX],data[alvo])
    pyplot.xlabel(labelEixoX)
    pyplot.ylabel(alvo)
    pyplot.show()
"""