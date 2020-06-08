#
#Todo o programa conta que o arquivo de codigo em c esta seguindo a convenção de boa programação em c e esta rodando
#
def trataLinha(linha,constantes,valConstantes,cFinal):#listas passadas por "referencia"(e um objeto mutavel então as alterações refletem externamente a função)
  if linha.count('//'):
    b=linha.split('//')
    b.pop()#joga fora o comentario
    linha=b.pop()#armazena o elemento string de novo em linha so que sem o comentario
                    
  if linha.count('#define'):#se for um define guarda os valores para substituir dps
    b = linha.split()
    b.remove('#define')
    for j in b:
        if j.isupper():
           constantes.append(j)
        else:
           valConstantes.append(j)
    linha = '\n'
  else:
      for c in constantes:
          if linha.count(c):
              linha = linha.replace(c,valConstantes[constantes.index(c)])
  cFinal.append(linha)

def removeFormatacao(linha,cFinal):
    aux = []
    if linha.count('printf("'):
        print()
    else:
      #linha.replace('  ', ' ')
      linha.replace('\t', ' ')
      aux = linha.split()
      linha = ""
      for i in aux:
          linha = linha + i
      cFinal.append(linha)

def concatenaLStrings(lista):
    aux = []
    for i in lista:
        aux = i.split()
        i=''
        for j in aux:
          if j:
            i=i+j+' '


arq = open('C:/Users/Leila/Desktop/trabED.txt','r')
codigoC = arq.readlines()
arq.close()
count = 0
const = []
valores = []
fCodigoC = []
final = []
for i in codigoC:
    trataLinha(i,const,valores,fCodigoC)
contador = fCodigoC.count('\n')
for i in fCodigoC:
  removeFormatacao(i,final)

if count:
  for i in range(count):
    fCodigoC.remove(i)
    
arq = open('saida.txt','w')
print(final)
final = concatenaLStrings(final)
arq.writelines(final)
arq.close()
