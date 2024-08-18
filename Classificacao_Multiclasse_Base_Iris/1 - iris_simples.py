import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # treinamento e teste
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder  # P/ não gerar o erro 1 e transformar o atributo categórico em numérico.
from keras.src.utils.np_utils import to_categorical  # transforma para poder identificar o neurônio de saída
from sklearn.metrics import confusion_matrix

"------------------Criação da rede neural"
base = pd.read_csv("\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Multiclasse_Base_Iris\\iris.csv")
"""
Nota, agora nossos atributos previsores e clase estão no mesmo arquivo.
"""

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
# iloc faz a divisão: iloc[intervalor de linhas, intervalo de colunas].values
# .values faz a conversão para formato numpy
# a função iloc não pega o intervalor superior (4)

# ----------------Corrigindo o erro 1
labelencoder = LabelEncoder()  # para problemas de classificação com mais de duas classes
classe = labelencoder.fit_transform(classe)  # converte as strings para valores numéricos
classe_dummy = to_categorical(classe)

"""
Análise para classificar nos neurônios de saída:
    Para resolver o problema de shape no neurônio de saída precisamos organizar os  dados conforme o exemplo abaixo. Se 
    fosse para 20 classes também seria assim.


Veja o que classe_dummy = to_categorical(classe) faz:
iris setosa       1 0 0  -> combinação para os 3 neurônios de saída
iris virginica    0 1 0  -> combinação para os 3 neurônios de saída
iris versicolor   0 0 1  -> combinação para os 3 neurônios de saída

Essas variáveis são chamadas do tipo 'dummy' depois que fazemos esse tipo de transformação.
Veja que agora agora classe e teste tem 3 dimensões.
Observe que o número de colunas da variável classe_dummy deve ter o mesmo comprimento do número de classes.
"""

# Divisão da base de dados entre treinamento e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy,
                                                                                              test_size=0.25)
# Criaçaõ da rede neural
classificador = Sequential()

# Primeira camada oculta + camada de entrada
classificador.add(Dense(units=4, activation='relu', input_dim=4))  # units=(entradas+saidas)/2 = (4+3)/2 = 4

# Segunda camada oculta
classificador.add(Dense(units=4, activation='relu'))

# camada de saída
classificador.add(Dense(units=3, activation='softmax'))
# activation='softmax' Para problemas de classificação com mais de duas classes. Gera probabilidade para cada classe e
# nos retorna o rótulo que possuir maior probabilidade

# Compilação da rede neural
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Adam: melhoria na descida do gradiente estocástica
# loss: 'categorical_crossentropy', pois agora temos três classes
# metrics: 'categorical_accuracy' para problemas de mais de duas classes, também podemos usar:
# 'kullback_leibler_divergence'

"------------------Treinamento da rede neural (o método fit)"


classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)
# epochs=1000: indica que vamos 1000 atualizações de pesos
# Erro 1: ValueError: Shapes (None, 1) and (None, 3) are incompatible
"""
Obs: essa acurácia: categorical_accuracy: 0.9732 está sendo realizada na própria base de dados de treinamento. Não é
real para julgar se o a rede neural está eficiente ou não.
"""

"-------------------Previsões"
resultado = classificador.evaluate(previsores_teste, classe_teste)
# método específico do Keras pra fazer a avaliação automática
# previsores_teste: São os 38 registros que será comparado com as respostas reais da classe teste
# temos aproximadamente 95 de acerto na base de dados de teste


# visualização da matriz de confusão (De forma manual)
previsoes = classificador.predict(previsores_teste)  # retorna a probabilidade
previsoes = (previsoes > 0.5)

# Retorna o índice que contém o maior valor para poder gerar a matriz de confusão
classe_teste2 = [np.argmax(t) for t in classe_teste]  # variável 't' foi criada agora
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)
print(matriz)
# muito útil para verificar quais classes estão acertando e errando mais
