##

import numpy as np
import matplotlib.pyplot as plt
import math


def graficos_linear_logistico(tipo=None):
	#função da reta
	# ax + by + c = 0

	#a, b e c são pesos que definem a posição da reta no plano
	a = -1
	b = 4
	c = 0.4

	#dado valor de X e Y, todo ponto sobre a reta tem que retornar 0
	# ax + by + c = 0
	# Se o valor retorna < zero está abaixo da reta
	# Se o valor retorna > zero está acima da reta
	# isso possibilita uma classificação linear

	#dados os pontos de X na reta, pode-se obter os valores de Y com a equação da reta
	#y = (-a*x -c)/b


	def linear(x):
	  y = (-a*x -c)/b
	  return y


    # função sigmoide
    # y = 1 / (1 + e ^ -x)
    # e -> número de Euler

    # a sigmoide sempre retornará um valor entre 0 e 1 
    # 0.001 e 0.999

	def sigmoid(x):
	    a = []
	    for item in x:
	        a.append(1/(1+math.exp(-item)))
	    return a



	plt.axvline(0, -1, 1, color='k', linewidth=1)
	plt.axhline(0, -2, 4, color='k', linewidth=1)

	# os Y serão definidos pelas funções
	if tipo == u'linear':
		x = np.linspace(-5, 5, 50)
		y = linear(x)	
	elif tipo == u'logistico':
		x = np.linspace(-10, 10, 50)
		y = sigmoid(x)
	
	plt.plot(x, y)
	plt.show(True)



def linear_salarios():
	x = np.linspace(0, 25, 5)
	y = np.linspace(20, 100, 5)
	#y = (-a*x -c)/b

	p1 = (3, 40)
	p2 = (10, 50)
	p3 = (20, 40)

	#achando os pesos da equação da reta pela determinante
	# pontos na reta
	# ponto 1 (0 20)
	# ponto 2 (25 100)
	# 0   20 1 0   20 
	# 25 100 1 25 100
	# x    y 1  x   y

	# -500 -0y -100x + 100 + 20x + 25y
	# -80x +24y - 400
    # DIVIDE POR 4

	a = -16
	b = 5
	c = -80
	# ax + by + c = 0
	# y = (-a*x -c)/b


	plt.axvline(0, -1, 1, color='k', linewidth=1)
	plt.axhline(0, -2, 4, color='k', linewidth=1)


	plt.plot(p1[0], p1[1], color='b', marker='o')
	plt.plot(p2[0], p2[1], color='r', marker='o')
	plt.plot(p3[0], p3[1], color='g', marker='o')
	plt.xlabel(u'Anos de Experiência')  
	plt.ylabel(u'Salários')

	plt.plot(x, y)

	ponto1 = (a*p1[0]) + (b*p1[1]) + c
	ponto2 = (a*p2[0]) + (b*p2[1]) + c
	ponto3 = (a*p3[0]) + (b*p3[1]) + c

	print("p1 %.2f" % (ponto1))
	print("p2 %.2f" % (ponto2))
	print("p3 %.2f" % (ponto3))

	plt.show(True)
  


#graficos_linear_logistico('linear')
graficos_linear_logistico('logistico')
linear_salarios()