import pandas as pd
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras import backend as k
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from sklearn.model_selection import cross_val_score

base = pd.read_csv("\\Users\\DELL\\PycharmProjects\\DeepLearning_DNN\\Classificacao_Multiclasse_Base_Iris\\iris.csv")
previsores = base.iloc[:, 0:4].values

classe = base.iloc[:, 4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = to_categorical(classe)


def criar_rede():
    k.clear_session()
    classificador = Sequential()
    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    classificador.add(Dense(units=4, activation='relu'))
    classificador.add(Dense(units=3, activation='softmax'))
    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return classificador


classificador = KerasClassifier(model=criar_rede, epochs=1000, batch_size=10)
resultados = cross_val_score(estimator=classificador,
                             X=previsores, y=classe_dummy,
                             cv=10, scoring='accuracy')

print(' ')
media = resultados.mean()
print(f'Acurácia: {media}')
desvio = resultados.std()
print(f'Desvio padrão: {desvio}')
