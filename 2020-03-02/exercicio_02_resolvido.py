import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class NeuralNetwork():
	dados = None
	rede = None
	teste_y = None
	previsoes_y = None

	def __init__(self, file=None, y=None):
		self.dados = pd.read_csv(file)

		#DIVIDINDO O DATASET ENTRE CARACTERÍSTICAS E CLASSIFICAÇÃO
		dados_x = self.dados.drop(y, axis=1)
		dados_y = self.dados[y]


		#DIVIDINDO MASSA DE DADOS PARA TREINAR O ALGORITMO (CARACTERÍSTICAS E CLASSIFICAÇÕES)
		#75% DOS DADOS PARA TREINAR E 25% DOS DADOS PARA TESTAR AS PREVISÕES
		SEED = 20
		np.random.seed = SEED

		treino_x, teste_x, treino_y, self.teste_y = train_test_split(dados_x, dados_y, test_size=0.25, stratify=dados_y)

		#CRIANDO A REDE NEURAL
		self.rede = MLPClassifier(hidden_layer_sizes=(20,20,20,20), max_iter=500, activation='relu', solver='lbfgs')
		self.rede.fit(treino_x, treino_y)

		#ANALISANDO AS previsoes_y DA REDE PARA OS DADOS DE TESTE
		self.previsoes_y = self.rede.predict(teste_x)
		self.menu()

	def graph(self):
		#impressão de um gráfico de compradores por cada característica
		seriesObj = self.dados.apply(lambda x: True if x['bought'] == 1 and x['home'] == 1 else False , axis=1)
		numOfRowsHome = len(seriesObj[seriesObj == True].index)

		seriesObj = self.dados.apply(lambda x: True if x['bought'] == 1 and x['how_it_works'] == 1 else False , axis=1)
		numOfRowsHow = len(seriesObj[seriesObj == True].index)

		seriesObj = self.dados.apply(lambda x: True if x['bought'] == 1 and x['contact'] == 1 else False , axis=1)
		numOfRowsContact = len(seriesObj[seriesObj == True].index)

		objects = ('home', 'how_it_works', 'contact')
		y_pos = np.arange(len(objects))
		performance = [numOfRowsHome,numOfRowsHow,numOfRowsContact]

		plt.bar(y_pos, performance, align='center', alpha=0.5)
		plt.xticks(y_pos, objects)
		plt.ylabel('Quantidade de pessoas')
		plt.title('Pessoas que compraram por página acessada')

		plt.show()

	def print_dados_originais(self):
		print(self.dados.head())
		print("\n\n###TOTAL DE LINHAS / COLUNAS####\n {}".format(self.dados.shape))
		print("\n\n###ESTATÍSTICAS DOS DADOS####\n {}".format(self.dados.describe().transpose()))


	def print_previsao_rede(self):
		print("\n\n###DADOS PARA TESTE DA REDE###")
		print("\n\nCompradores Reais: Total: {} \n Quantidades de Compradores: \n {} ".format(len(self.teste_y), self.teste_y.value_counts()))

		unique, counts = np.unique(self.previsoes_y, return_counts=True)
		print("\n\nPredição de Compradores pela Rede: Total: {} \n Quantidades de compradores: \n {} {}".format(len(self.previsoes_y), unique, counts))

		#VERIFICANDO A TAXA DE ACERTO DO ALGORITMO
		taxa_de_acerto = accuracy_score(self.teste_y, self.previsoes_y)
		print("\n\nTaxa de Acerto da Rede Neural: {}".format(taxa_de_acerto))


		print("\n\n###MATRIZ DE CONFUSÃO###")
		print(confusion_matrix(self.teste_y, self.previsoes_y))
		print(classification_report(self.teste_y, self.previsoes_y))


	def menu(self):

		while True:	
			os.system('clear')

			print("\n####REDE NEURAL PARA PREVISÃO DE COMPRAS POR NAVEGAÇÃO EM SITE#####")
			print("\n1 - Exibir dados originais e estatísticas")
			print("2 - Imprimir gráfico de Compras por página acessada")
			print("3 - Exibir previsão da Rede Neural")
			print("4 - Previsão para um usuário")
			opcao = int(input())

			if opcao == 1:
				self.print_dados_originais()
				input()
			if opcao == 2:
				self.graph()
			if opcao == 3:
				self.print_previsao_rede()
				input()
			if opcao == 4:
				print("\n\n####PREVISÃO PARA O USUÁRIO####")
				print("\nAcessou página inicial?")
				inicial = int(input())
				print("\nAcessou página de Como Funciona?")
				como_funciona = int(input())	
				print("\nAcessou página de contato?")
				contato = int(input())	
				usuario_x = [[inicial, como_funciona, contato]]
				previsao_usuario = self.rede.predict(usuario_x)

				if previsao_usuario[0] == 1:
					print("\nRESULTADO: Usuário irá efetuar compra.")
				else:
					print("\nRESULTADO: Usuário NÃO irá efetuar compra.")
				print("Pressione qualquer tecla...")
				input()
				os.system('clear')





rede = NeuralNetwork('tracking.csv', 'bought')
