import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pickle

"""
soft margin é a margem de quantos elementos 
podem ser "ignorados para se achar um 
hiperplano melhor
"""

def procurar_modelo_por_repeticao(X,y,repeticoes = 100,tamanho_teste = 0.3, kernel_do_modelo = "linear"):
    try:
        carregamentoPickle = open("modeloClassificacaoSVM_SVC.pickle", "rb")#<-carregando objeto do arquivo
        modeloRegressaoLinear = pickle.load(carregamentoPickle)   #<-carregando objeto do modelo previamente salvo
        x_treino, x_teste, y_treino, y_teste = sklearn.model_selection.train_test_split(X, y,test_size=tamanho_teste)  # separando a base para treinar o modelo e testar o mesmo
        melhorPrecisao = modeloRegressaoLinear.score(x_teste, y_teste)  # testando o modelo com dados não visto por ele ainda
        print("modelo carregado com sucesso")
    except:
        melhorPrecisao = 0 #flag para salvar o modelo de maior precisão
    for i in range(repeticoes):
        x_treino, x_teste, y_treino, y_teste = sklearn.model_selection.train_test_split(X, y, test_size=tamanho_teste)
        modeloClassificacao = svm.SVC(kernel=kernel_do_modelo)
        """     
        alterando a kernel do default para linear(conforme o tutorial)
        o resultado foi maior em uns 5%
        """
        modeloClassificacao.fit(x_treino,y_treino)
        predicao = modeloClassificacao.predict(x_teste)
        acuracia = metrics.accuracy_score(y_teste,predicao)
        if acuracia>melhorPrecisao:
            melhorPrecisao = acuracia
            with open("modeloClassificacaoSVM_SVC.pickle", "wb") as f:
                pickle.dump(modeloClassificacao, f)


dataCancer = datasets.load_breast_cancer()
X = dataCancer.data
y = dataCancer.target
classesAlvos = ['malignant', 'benign']

procurar_modelo_por_repeticao(X,y)
carregamentoPickle = open("modeloClassificacaoSVM_SVC.pickle", "rb")
modeloClassificacao = pickle.load(carregamentoPickle)

print("acuracia do modelo passando a base toda: ",modeloClassificacao.score(X,y))