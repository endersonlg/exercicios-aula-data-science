#BEM VINDO!!!

#ADICIONAR pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
#ADICIONAR pip install pandas

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

import pandas as pd;

dados = pd.read_csv("tracking.csv",);

treino = dados[['home','how_it_works','contact']];
resultado = dados['bought'];

treino_x = treino[0:75];
treino_y = resultado[0:75];

rede = MLPClassifier();

#METODO PARA TREINAR O ALGORITMO
rede.fit(treino_x, treino_y);

#TESTANDO PREVISOES DO ALGORITMO
teste_x = treino[75:100];
teste_y = resultado[75:100];

previsao_y = rede.predict(teste_x);
print(previsao_y);
print(accuracy_score(teste_y,previsao_y));
print(confusion_matrix(teste_y,previsao_y));