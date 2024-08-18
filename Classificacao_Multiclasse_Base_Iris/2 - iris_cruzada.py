import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.src.utils.np_utils import to_categorical

base = pd.read_csv("\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Multiclasse_Base_Iris\\iris.csv")
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    classificador.add(Dense(units=4, activation='relu'))
    classificador.add(Dense(units=3, activation='softmax'))
    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede, epochs=1000, batch_size=10)
resultados = cross_val_score(estimator=classificador,
                             X=previsores, y=classe_dummy,
                             cv=10, scoring='accuracy')

# Professor fez essa errada em não colocar a classe_dummy em y

media = resultados.mean()
desvio = resultados.std()
