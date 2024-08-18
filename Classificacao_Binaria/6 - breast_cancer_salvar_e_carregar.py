import keras
import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout

"""
Aprendendo a carregar e a salvar a rede neural, podendo ser usada em qualquer máquina.
"""

entradas_breast = "\\Users\\DELL\\PycharmProjects\\DeepLearning_DNN\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\DELL\\PycharmProjects\\DeepLearning_DNN\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)  # X
classe = pd.read_csv(saidas_breast)  # y


# Rede neural - Treinamento
classificador = Sequential()
classificador.add(Dense(units=12, activation='relu', kernel_initializer='normal', input_dim=30))
classificador.add(Dropout(0.3))
classificador.add(Dense(units=12, activation='relu', kernel_initializer='normal'))
classificador.add(Dropout(0.3))
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(previsores, classe, batch_size=10, epochs=100)
# Após o treinamento, a variável classificador já possui a estrutura e o conjunto dos pesos, mas não precisamos fazer
# isso todas as vezes. Para isso, vamos salvar esse classificador em disco.

# Salvando a rede
classificador.save('classificador_breast.keras')  # a extensão .keras é a padrão quando usamos o tensorflow

# Carregando o arquivo
classificador_novo = keras._tf_keras.keras.models.load_model("\\Users\\DELL\\PycharmProjects\\DNN\\"
                                                             "Classificacao_Binaria\\classificador_breast.keras")

# Estrutura
classificador_novo.summary()
print(' ')

# Classificação a partir do carregamento da rede neural
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005,
                  0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])
print(classificador_novo.predict(novo) > 0.95)
