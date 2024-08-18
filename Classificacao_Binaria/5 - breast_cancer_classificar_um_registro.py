import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout

"""
Quando a rede neural já estiver treinada, podemos fazer previsões/classficações. Se já sabemos quais são os melhores
parâmetros a serem usados, não precisamos fazer a divisão entre base de dados de treinamento e de teste, e também não 
há necessidade da realizar a validação cruzada, pois estas técnicas são usadas apenas para avaliar o algoritmo.

"""

entradas_breast = "\\Users\\DELL\\PycharmProjects\\DNN\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\DELL\\PycharmProjects\\DNN\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)  # X
classe = pd.read_csv(saidas_breast)  # y

# Veja que esses parâmetros vieram pós aplicação do Otimização com GridSearch. Esses são os parâmetros que retornam
# melhor resultado

classificador = Sequential()
classificador.add(Dense(units=12, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dropout(0.30))
classificador.add(Dense(units=12, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dropout(0.30))
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)  # Modelo da rede neural final com o conjunto de pesos

"""
Até aqui, será feito o treinamento para encontrar os melhores pesos. Note que aqui não estamos mais usando nossa base de 
dados de teste. Este é o nosso modelo final.
"""

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005,
                  0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

""" 
Dois colchetes, o primeira significa linha e o segundo as colunas. Precisamos dos 30 atributos da tabela. Esses 
valores são aleatórios. 
Precisamos desses 30 atributos pois são as entradas para a rede neural
"""

print(' ')
previsao = classificador.predict(novo)
# Parece com o que fizemos com a base de dados de teste, mas aqui temos apenas um registro
print(previsao)
print(' ')

previsao = (previsao > 0.95)  # vai ser True se a previsão foir maior que 0.95
print(previsao)

"""
O valor da previsão foi igual a 1, ou seja, com aqueles dados de entradas chegamos a conclusão que é um tumor do tipo 
maligno. 
"""
