import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV  # Realiza a tunagem dos hiperparâmetros
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras import backend as k  # trabalha com o conceito de seções
from keras._tf_keras.keras.layers import Dense, Dropout

"""
A tunagem dos hiperparâmetro encontra a melhor configuração para o modelo. É um teste exaustivo que combina várias
possibilidades.
"""

entradas_breast = "\\Users\\DELL\\PycharmProjects\\DeepLearning_DNN\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\DELL\\PycharmProjects\\DeepLearning_DNN\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)  # X
classe = pd.read_csv(saidas_breast)  # y


def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):

    k.clear_session()
    classificador = Sequential()

    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classificador.add(Dropout(0.3))  # 30% dos neurônios da camada de entrada serão zerados

    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.3))  # 30% dos neurônios da camada oculta serão zerados

    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return classificador


# Variável classificador
classificador = KerasClassifier(model=criar_rede)
# model: modelo que recebe a a função que cria a rede neural

# Dicionário de parâmetros
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'model__optimizer': ['adam', 'sgd'],
              'model__loss': ['binary_crossentropy', 'hinge'],
              'model__kernel_initializer': ['random_uniform', 'normal'],
              'model__activation': ['relu', 'tanh'],
              'model__neurons': [16, 8]}


"""
Nota:
Combinações que o GridSearch irá realizar: 2 * 2 * 2 * 2 * 2 * 2 * 2 = 2^7 = 128 combinações
"""

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)
# cv: cross validation. No scrip anterios escolhemos K=10, mas aqui para agilizar vamos escolher K=5. Portanto,
# a quantidade de combinações total é: 2^7 * 5 = 640

grid_search = grid_search.fit(X=previsores, y=classe)

print(grid_search)  # retorna os melhores parâmetros
print(' ')

melhores_parametros = grid_search.best_params_
print(melhores_parametros)
