import keras._tf_keras.keras.optimizers
import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score  # Função que faz a divisão da base de dados (Cross Validation)
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras import backend as k  # trabalha com o conceito de seções
from keras._tf_keras.keras.layers import Dense, Dropout


"""
- VALIDAÇÃO CRUZADA (k-FOLD CROSS VALIDATION). Uma técnica mais eficiente para fazer avaliações de algoritmos de 
aprendizagem de máquinas. Em trabalhos científicos e de pesquisa essa técnica é a mais utilizada. Com essa prática, 
todos os modelos são usados para fazer treinamento e teste de maneira alternada. Ou seja, aqui, todos os dados são
usados para treinamento e teste (não tem separação da base de dados em treinamento e teste).

- O percentual de acerto é a média da precisão obtida em cada K (cv).

- K = 10, muito aceito na comunidade científica.

Modelos ruins
- underfitting: Seria como tentar eliminar um tiranossaura-rex (problema complexo) com uma raquete
    - Terá resultados ruins na base de treinamento
    - A rede não consegue explorar os dados com eficácia
- overfitting: Seria como tentar matar um mosquito (problema simples) com uma bazuca (muitos recursos)
    - Terá resultados bons na base de dados de treinamento
    - Terá resultados ruins na base de dados de teste
    - Muito específico
    - Memorização
    - Erros na variação de novas instânciasModelo bom
    
Modelo bom:
Um problema complexo deve ser resolvido com um modelo complexo. Um problema simples deve ser resolvido com um modelo
simples.
    
Dropout: para corrigir ou atenuar o problema de overfitting. Irá zerá alguns valores da entrada (na camada de entrada ou 
camada oculta), para que esses valores (aleatórios) não tenham influencia no resultado final.

"""
entradas_breast = "\\Users\\DELL\\PycharmProjects\\DNN\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\DELL\\PycharmProjects\\DNN\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)
classe = pd.read_csv(saidas_breast)


def criar_rede():  # Para a validação cruzada precisamos criar uma função

    k.clear_session()  # limpa a seção do tensorflow
    classificador = Sequential()

    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    classificador.add(Dropout(0.3))  # 30% dos neurônios da camada de entrada serão zerados
    """
     Nota: dropout: sempre após a criação de uma camada. Serve para zerar alguns neurônios de forma que estes não tenham 
     nenhuma influência no resultado final. É recomendado ter um dropout entre 20 e 30% 
     
     Artigo: Dropout: A Simple Way to Prevent Neural Networks from Overfitting (20214)
     """
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.3))  # 30% dos neurônios da camada oculta serão zerados

    classificador.add(Dense(units=1, activation='sigmoid'))

    otimizador = keras._tf_keras.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return classificador


# Variável classificador
classificador = KerasClassifier(model=criar_rede, epochs=100, batch_size=10)
# model: modelo que recebe a a função que cria a rede neural

# Realiza os testes. Teremos 10 resultados. Efetivamente fará a validação cruzada
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')
print(f'Avaliação de cada fatia da validação cruzada: {resultados}')

# X: indica quais são os atributos previsores
# y: recebe a classe
# cv: faz a divisão da base de dados para fazer a validação cruzada. É o K, nesse caso teremos 10 fatias, sendo que uma
# delas será usada para teste enquanto o restante (90%) será usado para treinamento. Vamos alternar entre todas as
# fatias para fazer a validação cruzada
# scoring: a forma como queremos retornar o resultado

# Daqui pra cima ele já realiza a o treinamento com a validação cruzada

print(' ')
media = resultados.mean()  # média, para saber o percentual de acerto da base de dados
print(f'Média: {media}')

print(' ')
desvio = resultados.std()  # desvio padrão, informa o quanto cada resultado da fatia de K se distancia da média
print(f'Desvio padrão: {desvio}')
# desvio padrão, quanto maior o desvio, maior a chance de ter overfitting na base de dados
# overfitting: Quando o algoritmo (rede neural) se adapta demais a base de dados de treinamento. Isso implica que
# quando vamos passar dados novos para essa rede, ela não vai nos fornecer bons resultados
