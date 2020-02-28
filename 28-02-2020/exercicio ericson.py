import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("tracking.csv")

X = df[['home','how_it_works','contact']]
#print(X);
y = df['bought']

treino_x,teste_x,treino_y,teste_y = train_test_split(X,y,test_size = 0.25)
print(treino_x)

rede = MLPClassifier()

rede.fit(treino_x,treino_y)

previsao_y = rede.predict(teste_x)
# print(previsao_y)

# print(accuracy_score(teste_y,previsao_y)*100)