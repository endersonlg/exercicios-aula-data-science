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

  def __init__(self,file=None):
    self.df = pd.read_csv(file)

    # INSERINDO NOME NAS COLUNAS
    self.df.columns = ['Produtor',
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
    print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.df.shape))
    print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.df.describe().transpose()))
    print("\n\n###5 PRIMEIROS DADOS####\n {}".format(self.df.head()))
              
    self.grafico()
    self.redeNeural()

  def grafico(self):
    # GRAFICO
    x = np.linspace(11,15,20)
    y = np.linspace(80,160,20)

    self.df.plot(kind='scatter',x='Alcool',y='Magnésio',color=self.df['Produtor'],s=self.df['Fenóis totais']*50)

    plt.show(True)

  def redeNeural(self):
    # REDE NEURAL

    self.rede = MLPClassifier()
    
    SEED = 20
    np.random.seed = SEED


    produtor1 = self.df.loc[(self.df['Produtor']) == 1]
    produtor2 = self.df.loc[(self.df['Produtor']) == 2]
    produtor3 = self.df.loc[(self.df['Produtor']) == 3]
      
    produtor1_x = produtor1.drop('Produtor',axis=1)
    produtor1_y = produtor1['Produtor']

    produtor2_x = produtor2.drop('Produtor',axis=1)
    produtor2_y = produtor2['Produtor']

    produtor3_x = produtor3.drop('Produtor',axis=1)
    produtor3_y = produtor3['Produtor']

    treino_produtor1_x,teste_produtor1_x,treino_produtor1_y,teste_produtor1_y = train_test_split(produtor1_x,produtor1_y,test_size=0.25,stratify=produtor1_y)
    treino_produtor2_x,teste_produtor2_x,treino_produtor2_y,teste_produtor2_y = train_test_split(produtor2_x,produtor2_y,test_size=0.25,stratify=produtor2_y)
    treino_produtor3_x,teste_produtor3_x,treino_produtor3_y,teste_produtor3_y = train_test_split(produtor3_x,produtor3_y,test_size=0.25,stratify=produtor3_y)
      
    self.rede.fit(treino_produtor1_x,treino_produtor1_y)
    self.rede.fit(treino_produtor2_x,treino_produtor2_y)
    self.rede.fit(treino_produtor3_x,treino_produtor3_y)

    produtor1_previsao_y = self.rede.predict(teste_produtor1_x)
    produtor2_previsao_y = self.rede.predict(teste_produtor2_x)
    produtor3_previsao_y = self.rede.predict(teste_produtor3_x)


    print("###PREVISÃO DO RESULTADO PRODUTOR 1###\n {}\n".format(produtor1_previsao_y))
    print("###TAXA DE ACERTO PRODUTOR 1###\n {}%\n".format(accuracy_score(teste_produtor1_y, produtor1_previsao_y)*100))
    print("###MATRIZ DE CONFUSÃO PRODUTOR 1### \n {}\n".format(confusion_matrix(teste_produtor1_y, produtor1_previsao_y)))

    print("###PREVISÃO DO RESULTADO PRODUTOR 2###\n {}\n".format(produtor2_previsao_y))
    print("###TAXA DE ACERTO PRODUTOR 2###\n {}%\n".format(accuracy_score(teste_produtor2_y, produtor2_previsao_y)*100))
    print("###MATRIZ DE CONFUSÃO PRODUTOR 2### \n {}\n".format(confusion_matrix(teste_produtor2_y, produtor2_previsao_y)))

    print("###PREVISÃO DO RESULTADO PRODUTOR 3###\n {}\n".format(produtor3_previsao_y))
    print("###TAXA DE ACERTO PRODUTOR 3###\n {}%\n".format(accuracy_score(teste_produtor3_y, produtor3_previsao_y)*100))
    print("###MATRIZ DE CONFUSÃO PRODUTOR 3### \n {}\n".format(confusion_matrix(teste_produtor3_y, produtor3_previsao_y)))
   
rede = NeuralNetwork("wine.csv")