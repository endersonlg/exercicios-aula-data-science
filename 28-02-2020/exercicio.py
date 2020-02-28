#BEM VINDO!!!

#ADICIONAR pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
#ADICIONAR pip install pandas

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import pandas as pd;

dados = pd.read_csv("tracking.csv",);

treino_x = dados[['home','how_it_works','contact']];
treino_y = dados['bought'];

rede = MLPClassifier();

#METODO PARA TREINAR O ALGORITMO
# rede.fit(treino_x, treino_y);

