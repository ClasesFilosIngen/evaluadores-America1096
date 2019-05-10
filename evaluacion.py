def obtenerV(datosEntrenamiento):
	Voc, C, V = {}, {}, {}
	j, k = 1, 0
	ln = []

	for datos in datosEntrenamiento:
		[Voc.setdefault(x,0) for x in datos]

	for datos in datosEntrenamiento:
		cAux = []
		for i in datos:
			if i not in ln:
				ln.append(i)
				Voc[i] = j
				j = j+1
				
			cAux.append(Voc[i])
		C.setdefault(k, cAux)
		k = k + 1

	[V.setdefault(Voc[key],0) for key in Voc]
	
	for datos in range(2):
		for key in Voc:
			V.update({Voc[key]:V.get(Voc[key])+datosEntrenamiento[datos].count(key)})

	return (C, V, Voc)

def transformar(Voc, w):
	for key in Voc:
		if key == w:
			return Voc[key]
	return 0

def naiveBayes(C, V, Voc, M):
	resultados = []
	tam = []
	k = 1
	for key in C:
		tam.append(len(C[key]))

	resultados.append((tam[0]) / (tam[0] + tam[1]) )
	resultados.append((tam[1]) / (tam[0] + tam[1]) )

	pMC0 = resultados[0]
	pMC1 = resultados[1]

	for w in M:
		pMC0 = pMC0 * (C[0].count(w)) / (tam[0])
		pMC1 = pMC1 * (C[1].count(w)) / (tam[1])

	resultados.append(pMC0)
	resultados.append(pMC1)
	
	resultados.append(pMC0 + pMC1)
	resultados.append(pMC0/resultados[len(resultados)-1])
	resultados.append(pMC1/resultados[len(resultados)-2])

	return resultados


def clasificadorBayes(C, V, Voc, M):
	resultados = []
	tam = []
	k = 1
	for key in C:
		tam.append(len(C[key]))

	resultados.append((tam[0] + k) / (tam[0] + tam[1] + k * 2) )
	resultados.append((tam[1] + k) / (tam[0] + tam[1] + k * 2) )

	pMC0 = resultados[0]
	pMC1 = resultados[1]

	for w in M:
		pMC0 = pMC0 * (C[0].count(w) + k)/(tam[0] + k * len(V))
		pMC1 = pMC1 * (C[1].count(w) + k)/(tam[1] + k * len(V))

	resultados.append(pMC0)
	resultados.append(pMC1)
	
	resultados.append(pMC0 + pMC1)
	resultados.append(pMC0/resultados[len(resultados)-1])
	resultados.append(pMC1/resultados[len(resultados)-2])

	return resultados

def mostrarDatos(datosEntrenamiento, M):
	tr = []

	(C, V, Voc) = obtenerV(datosEntrenamiento)

	for w in M:
		tr.append(transformar(Voc, w))

	print("M es", tr)
	
	for key in C:
		print("C", key, " =", C[key])

	print("Vocabulario =", V)
	print(Voc)
	res1 = naiveBayes(C, V, Voc, tr)
	res2 = clasificadorBayes(C, V, Voc, tr)

	print("Naive Bayes\t\t\t\tAlisando")
	print("P(C = C0) =", res1[0], "\t\t\tP(C = C0) =", res2[0])
	print("P(C = C1) =", res1[1], "\t\t\tP(C = C1) =", res2[1])
	print("P(M, C = C0) =", res1[2], "\t\t\tP(M, C = C0) =", res2[2])
	print("P(M, C = C1) =", res1[3], "\tP(M, C = C1) =", res2[3])
	print("P(M) =", res1[4], "\t\tP(M) =", res2[4])
	print("P(C = C0 | M) =", res1[5], "\t\t\tP(C = C0 | M) =", res2[5])
	print("P(C = C1 | M) =", res1[6], "\t\t\tP(C = C1 | M) =", res2[6])

def datosReales(s, h, datosEntrenamiento):
	tr = []

	(C, V, Voc) = obtenerV(datosEntrenamiento)

	spam, ham = [], []
	spamR, hamR = [], []
	for w in s:
		spam.append(transformar(Voc, w))

	for w in h:
		ham.append(transformar(Voc, w))

	for wt in spam:
		resSp = clasificadorBayes(C, V, Voc, [wt])
		c0 = resSp[len(resSp)-2]
		c1 = resSp[len(resSp)-1]
		if c0 > c1:
			spamR.append(wt)
		else:
			hamR.append(wt)

	for wt in ham:
		resSp = clasificadorBayes(C, V, Voc, [wt])
		c0 = resSp[len(resSp)-2]
		c1 = resSp[len(resSp)-1]
		if c0 > c1:
			spamR.append(wt)
		else:
			hamR.append(wt)				

	return (spam, ham, spamR, hamR)
	

def matrizDeConfusion(s, h, datosEntrenamiento):
	m = [[None] * 2 for i in range(2)]
	(spam, ham, spamR, hamR) = datosReales(s, h, datosEntrenamiento)

	print("spam esperado =", spam)
	print("ham esperado =", ham)
	print("spam obtenido=", spamR)
	print("ham obtenido =", hamR)

	tp, tn, fp, fn = 0, 0, 0, 0
	
	for s1 in spam:
		if s1 in spamR:
			if spam.count(s1) == spamR.count(s1):
				tp = tp + 1
			elif spamR.count(s1)-ham.count(s1) == spam.count(s1):
				tp = tp + 1
		if s1 in hamR:
			if spam.count(s1) == hamR.count(s1):
				fp = fp + 1
			elif hamR.count(s1)-ham.count(s1) == spam.count(s1):
				fp = fp + 1

	for h1 in ham:
		if h1 in hamR:
			if ham.count(h1) == hamR.count(h1):
				tn = tn + 1
			elif hamR.count(h1)-spam.count(h1) == ham.count(h1):
				tn = tn + 1
		if h1 in spamR:
			if ham.count(h1) == spamR.count(h1):
				fn = fn + 1
			elif spamR.count(h1)-spam.count(h1) == ham.count(h1):
				fn = fn + 1

	m[0][0] = tp
	m[0][1] = fp
	m[1][0] = fn
	m[1][1] = tn

	return m

def evaluacion(spam, ham, datosEntrenamiento):
	m = matrizDeConfusion(spam, ham, datosEntrenamiento)
	resultados = []

	resultados.append((m[0][0] + m[1][1]) / (m[0][0] + m[0][1] + m[1][0] + m[1][1]))
	resultados.append(m[0][0] / (m[0][0] + m[0][1]))
	resultados.append(m[0][0] / ( m[0][0] + m[1][0]))
	resultados.append((2*resultados[1]*resultados[2]) / (resultados[1]+resultados[2]))

	return resultados

def mostrarEvaluacion(spam, ham, datosEntrenamiento):

	resultados = evaluacion(spam, ham, datosEntrenamiento)
	print("\n\t*** Evaluacion del Clasificador de Bayes ***")
	print("\t    - Accuracy =", resultados[0])
	print("\t    - Precision =", resultados[1])
	print("\t    - Recall =", resultados[2])
	print("\t    - F1-score =", resultados[3])

def main():
	datosEntrenamiento = [
							["oferta", "es", "secreto", 
							"click", "link", "secreto", 
							"secreto", "deportes", "link"],
							["practica", "deportes", "hay",
							"fue", "practica", "deportes",
							"secreto", "deportes", "evento",
							"deportes", "es", "hay",
							"deportes", "cuesta", "dinero", "link"] 
						]

	spam = ["secreto", "deportes", "link", "es"]
	ham = ["practica", "evento", "dinero", "deportes", "link"]


	mostrarEvaluacion(spam, ham, datosEntrenamiento)



main()