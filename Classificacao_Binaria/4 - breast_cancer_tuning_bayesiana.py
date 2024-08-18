import pandas as pd
from sklearn.model_selection import cross_val_predict
from skopt import gp_minimize
from skopt.space import Integer, Categorical

from keras._tf_keras.keras.layers import Dense, Dropout
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from keras._tf_keras.keras.models import Sequential


entradas_breast = "\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Binaria\\entradas_breast.csv"
saidas_breast = "\\Users\\franc\\PycharmProjects\\NeuralNetworks\\Classificacao_Binaria\\saidas_breast.csv"
previsores = pd.read_csv(entradas_breast)
classe = pd.read_csv(saidas_breast)


def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
    model = Sequential()
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    model.add(Dropout(0.2))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# Classe personalizada para o classificador Keras
class KerasClassifierWrapper(BaseEstimator):
    def __init__(self, optimizer='adam', loss='binary_crossentropy', kernel_initializer='random_uniform',
                 activation='relu', neurons=16, epochs=100, batch_size=10):
        self.optimizer = optimizer
        self.loss = loss
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.neurons = neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = criar_rede(self.optimizer, self.loss, self.kernel_initializer,
                                self.activation, self.neurons)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

space = [
    Integer(8, 16, name='neurons'),
    Categorical(['adam', 'sgd'], name='optimizer'),
    Categorical(['relu', 'tanh'], name='activation'),
    Categorical(['random_uniform', 'normal'], name="kernel_initializer"),
    Categorical(['binary_crossentropy', 'hinge'], name="loss")]

# FO para otimização
def objective(params):
    neurons, optimizer, activation, kernel_initializer, loss = params
    predictions = cross_val_predict(estimator=KerasClassifierWrapper(optimizer=optimizer,
                                                                     loss=loss,
                                                                     kernel_initializer=kernel_initializer,
                                                                     activation=activation, neurons=neurons),
                                                                     X=previsores, y=classe, cv=10)
    accuracy = accuracy_score(classe, predictions)
    return -accuracy  # Remova o sinal negativo, pois estamos maximizando a acurácia

result = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=0)

# Melhores hiperparâmetros encontrados
best_params = {
    'neurons': result.x[0],
    'optimizer': result.x[1],
    'activation': result.x[2],
    'kernel_initializer': result.x[3],
    'loss': result.x[4]
}
print("Melhores hiperparâmetros encontrados: ")
print(best_params)
