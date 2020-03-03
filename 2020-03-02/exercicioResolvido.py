##

# ADICIONAR pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
# ADICIONAR pip install pandas

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class NeuralNetwork():
    df = None
    rede = None

    def __init__(self, file=None):
        self.df = pd.read_csv(file)
        # print(self.df.head())
        # print(self.df.shape)
        # print(self.df.describe().transpose())
        print(self.df.bought.value_counts())
        treino = self.df[:25]
        print(treino.bought.value_counts())
        SEED = 20
        np.random.seed = SEED

        x = self.df.drop('bought', axis=1)
        y = self.df.bought

        treino_x, teste_x, treino_y, teste_y = train_test_split(
            x, y, test_size=0.25, stratify=y)

        # print(teste_x.value_counts())
        # print(teste_y.value_counts())

        self.rede = MLPClassifier()
        self.rede.fit(treino_x, treino_y)

        previsoes_y = self.rede.predict(teste_x)

        print(accuracy_score(teste_y, previsoes_y)*100)
        print(confusion_matrix(teste_y, previsoes_y))

        self.menu()

    def menu(self):
        while(True):
            os.system("cls")
            print("####PREVISÂO DE COMPRAS POR USUÁRIO####")
            print("Usuário acessou a página home(0-1)?")
            home = int(input())
            print("Usuário acessou a página como funciona(0-1)?")
            how = int(input())
            print("Usuário acessou a página contato(0-1)?")
            contact = int(input())
            previsao = self.rede.predict([[home, how, contact]])
            print("ESSE USUÁRIO COMPRA OU NÂO? ->{}".format(previsao))
            print("Pressione qualquer tecla...")
            input()


rede = NeuralNetwork("tracking.csv")
