import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # treinamento e teste
from keras._tf_keras.keras import Sequential
from keras._tf_keras.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder  # P/ não gerar o erro 1 e transformar o atributo categórico em numérico.
from keras._tf_keras.keras.utils import to_categorical  # transforma para poder identificar o neurônio de saída
from sklearn.metrics import confusion_matrix, accuracy_score


"------------------Criação da rede neural"
base = pd.read_csv("\\Users\\DELL\\PycharmProjects\\DeepLearning_DNN\\Classificacao_Multiclasse_Base_Iris\\iris.csv")

"""
Nota, agora nossos atributos previsores e clase estão no mesmo arquivo.
"""

previsores = base.iloc[:, 0:4].values  # a função iloc não pega o intervalor superior (4)
classe = base.iloc[:, 4].values
# iloc faz a divisão: iloc[intervalor de linhas, intervalo de colunas].values
# .values faz a conversão para formato numpy array

"""
Observe que agora, as classes são strings, e a assim a rede neural não irá interpretá-los para fazer os cálculos.
"""


# ---------------- Transformandos a classe
labelencoder = LabelEncoder()  # para problemas de classificação com mais de duas classes
classe = labelencoder.fit_transform(classe)  # converte as strings em valores numéricos
classe_dummy = to_categorical(classe)  # Cria uma sequência para cada uma das 3 saídas

"""
Veja como a rede neural irá classificar os neurônios de saída (se fosse para 20 classes também seria assim):


Veja o que classe_dummy = to_categorical(classe) faz:

iris setosa       1 0 0  -> combinação para os 3 neurônios de saída
    - Se houver uma alta probabilidade para o primeiro neurônio e baixa para os outros, será classificado como 
    iris setosa.
    
iris virgínica    0 1 0  -> combinação para os 3 neurônios de saída
    - Se houver uma alta probabilidade para o segundo neurônio e baixa para os outros dois, será classificado com 
    iris virgínica.
    
iris versicolor   0 0 1  -> combinação para os 3 neurônios de saída
    - Se houver uma alta probabilidade para o terceiro neurônio e baixa para os outros, será classificado com 
    iris versicolor.
    
Essas variáveis são chamadas do tipo 'dummy' depois que fazemos esse tipo de transformação.
Veja que agora agora classe e teste tem 3 dimensões.
Observe que o número de colunas da variável classe_dummy deve ter o mesmo comprimento do número de classes de saída
(no caso 3).
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
# activation='softmax' para problemas de classificação com mais de duas classes. Gera probabilidade para cada classe e
# nos retorna o rótulo que possuir maior probabilidade

# Compilação da rede neural
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Adam: melhoria na descida do gradiente estocástica
# loss: 'categorical_crossentropy', pois agora temos três classes
# metrics: 'categorical_accuracy' para problemas com mais de duas classes, também podemos usar:
# 'kullback_leibler_divergence'

"------------------Treinamento da rede neural (o método fit)"


classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)
# epochs=1000: indica que vamos 1000 atualizações de pesos
# ---------------------------------------------------Erro 1: ValueError: Shapes (None, 1) and (None, 3) are incompatible
"""
Obs: essa acurácia: categorical_accuracy: 0.9841 está sendo realizada na própria base de dados de treinamento. Ou seja,
não é real, não podemos julgar se a rede neural está eficiente ou não.
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

# Taxa de acerto
print(' ')
print(f'Acurácia real da rede neural: {accuracy_score(classe_teste2, previsoes2)}')
