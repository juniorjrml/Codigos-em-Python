arq = open('teste1.txt', 'r')

estado_inicial = arq.readline().split()
estados_finais = arq.readline().split()

print("\n\n\tTRADUÇÃO DE AUTOMATO PARA GRAMÁTICA")
print("\n\t\t\tESTADO INICIAL = ", estado_inicial[0])

try:
    while (arq):
    	trad = arq.readline().split()
    	print("\t\t\t", trad[0], "-->", trad[1] + trad[2])
except IndexError:
    pass

for i in range(len(estados_finais)):
    print("\t\t\t", estados_finais[i], "--> Epsilon\n", end="")

arq.close
