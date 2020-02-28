#BEM VINDO!!!

#ADICIONAR pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

porco1 = [0, 0, 1];
porco2 = [0, 1, 1];
porco3 = [1, 0, 1];

cachorro1 = [1, 1, 1];
cachorro2 = [1, 1, 0];
cachorro3 = [1, 0, 0];

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3];
treino_y = [1, 1, 1, 0, 0, 0];
rede = MLPClassifier();

#METODO PARA TREINAR O ALGORITMO
rede.fit(treino_x, treino_y);

#TESTANDO PREVISOES DO ALGORITMO
teste_x = [[0, 0, 0],[0, 1, 0]];
teste_y = [1,1]

previsao_y = rede.predict(teste_x);
print(teste_y);
print(previsao_y);
print(accuracy_score(teste_y,previsao_y));
print(confusion_matrix(teste_y,previsao_y));
