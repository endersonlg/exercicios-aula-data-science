import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


class NeuralNetwork():
    df = None
    rede = None

    def __init__(this, file=None):
        this.df = pd.read_csv(file)

        # INSERINDO NOME NAS COLUNAS
        this.df.columns = ['Produtor',
                           'Alcool',
                           'Ácido málico',
                           'Cinza',
                           'Alcalinidade das cinzas',
                           'Magnésio',
                           'Fenóis totais',
                           'Flavonóides',
                           'Fenóis não flavonóides',
                           'Proantocianinas',
                           'Intensidade da cor',
                           'Matiz',
                           'OD315 de vinhos diluídos ',
                           'Prolina']

        # MOSTRA DETALHES DA TABELA
        print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(this.df.shape))
        print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(
            this.df.describe().transpose()))
        print("\n\n###5 PRIMEIROS DADOS####\n {}".format(this.df.head()))

        # GRAFICO
        x = np.linspace(11, 15, 20)
        y = np.linspace(80, 160, 20)

        this.df.plot(kind='scatter', x='Alcool', y='Magnésio',
                     color=this.df['Produtor'], s=this.df['Fenóis totais']*50)

        plt.show(True)

        # REDE NEURAL

        rede = MLPClassifier()

        SEED = 20
        np.random.seed = SEED

        x = this.df.drop('Produtor', axis=1)
        y = this.df.Produtor

        treino_x, teste_x, treino_y, teste_y = train_test_split(
            x, y, test_size=0.25, stratify=y)

        rede.fit(treino_x, treino_y)

        previsao_y = rede.predict(teste_x)

        print("\n\n###DADOS UTILIZADOS####\n {}\n".format(treino_x))

        print("###PREVISÃO DO RESULTADO###\n {}\n".format(previsao_y))

        print("###TAXA DE ACERTO###\n {}%\n".format(
            accuracy_score(teste_y, previsao_y)*100))

        print("###MATRIZ DE CONFUSÃO### \n {}\n".format(
            confusion_matrix(teste_y, previsao_y)))


rede = NeuralNetwork("wine.csv")
