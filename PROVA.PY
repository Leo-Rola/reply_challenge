import math
import numpy
import random

MATRICE=[]
COLONNE=0
RIGHE=0
NUMERO_SERPENTI=0
SERPENTI=[]

def take_data(nome_file):
    global COLONNE
    global RIGHE
    global NUMERO_SERPENTI
    global MATRICE
    global SERPENTI
    f = open(nome_file, "r")
    intestazione = f.readline().split()
    COLONNE = int(intestazione[0])
    RIGHE = int(intestazione[1])
    NUMERO_SERPENTI = int(intestazione[2])
    SERPENTI = [int(i) for i in f.readline().split()]
    print(f"serpenti = {SERPENTI}")
    for line in f:
        line = line.split()
        linetemp=[]
        for j in range(len(line)):
            if  line[j]=='*':
                linetemp.append(0.5)
            elif line[j].lstrip('-').isdigit():
                linetemp.append(int(line[j]))

        MATRICE.append(linetemp)
    f.close()
    
def print_data_output(matrix, file):
    file = open(file, "w")
    stringTemp=""
    for r in matrix:
        for j in range(len(r)):
            if type(r[j]).__name__=='float':
                r[j]=int(r[j])
        r=[str(y) for y in r]
        for i in r:
            stringTemp=stringTemp+i+" "

        file.write(stringTemp)
        file.write("\n")
        print(stringTemp)

    file.close
    
def compute_fitness(chosen_snakes):
    global COLONNE
    global RIGHE
    global NUMERO_SERPENTI
    global MATRICE
    global SERPENTI
    pos=()
    error=False
    reward=0
    flag_matrix=numpy.ones((RIGHE,COLONNE), dtype=bool)
    for I,snake in enumerate(chosen_snakes):
        if error==True:
            reward=0
            break
        for i,charr in enumerate(snake):
            if charr.isdigit() and snake[i+1].isdigit(): 
                if i==0:
                    if MATRICE[int(snake[0])][int(snake[1])]==0.5:
                        chosen_snakes[I][i]=str(int(chosen_snakes[I][i])+1)
                        error("errore: warmhole at the beginning")
                        reward=compute_fitness(chosen_snakes)
                        error=True
                        break
                if i==(len(snake)-2):
                    if MATRICE[int(snake[i])][int(snake[i+1])]==0.5:
                        error=True
                        break
                pos=(int(charr),int(snake[i+1]))
                if flag_matrix[int(charr)][int(snake[i+1])]==False:
                    error=True
                    break
                flag_matrix[int(charr)][int(snake[i+1])]=False
                reward+=MATRICE[int(charr)][int(snake[i+1])]

            if not(charr.isdigit()):
                if charr =="U":
                    pos=((pos[0]-1)%RIGHE,pos[1])
                if charr =="D":
                    pos=((pos[0]+1)%RIGHE,pos[1])
                if charr=="L":
                    pos=(pos[0],(pos[1]-1)%COLONNE)
                if charr=="R":
                    pos=(pos[0],(pos[1]+1)%COLONNE)
                
                if flag_matrix[pos[0]][pos[1]]==False:
                    error=True
                    break
                if i==(len(snake)-1):
                    if MATRICE[pos[0]][pos[1]]==0.5:
                        error=True
                        break
                if MATRICE[pos[0]][pos[1]]==0.5:
                    if not(snake[i+1].isdigit()):
                        error=True
                        break
                    if snake[i+1].isdigit() and MATRICE[snake[i+1]][snake[i+2]]!=0.5:
                        error=True
                        break
                else:
                    reward+=MATRICE[pos[0]][pos[1]]
                    flag_matrix[pos[0],pos[1]]=False

    return reward
    
    
def genera_percorso( serpente, indice ):
        stringa = ["U", "D", "L", "R"]
        percorso = []
        percorso.append( str(lista_finestre_sorted[indice][0][0]) )
        percorso.append( str(lista_finestre_sorted[indice][0][1]) )
            
        for i in range(serpente):
            pos= random.randint(0,3)
            percorso.append(stringa[pos])
        print(percorso)
        return percorso


if __name__ == '__main__':
    take_data("02-swarming-ant.txt")
        
    sommaSerpenti = sum(SERPENTI)
    print(f"somma serpenti = {sommaSerpenti}")
    lunghezzaMedia = sommaSerpenti/ len(SERPENTI)
    print(f"lunghezza media = {lunghezzaMedia}")
    lato = int( math.sqrt(lunghezzaMedia) )
    print(f"lato {lato}")

    lista_finestre = list()
    for r in range(0, RIGHE - (lato-1) , lato-1):
        for c in range(0, COLONNE - (lato -1), lato-1):
            
            init_pos = ( int(r+lato/2), int(c+lato/2))
            sommaValoriFinestra = 0
            for i in range(lato): #colonne
                    
                for j in range(lato):
                    
                    sommaValoriFinestra += MATRICE[r+i][c+j] 
            
            tupla = (init_pos, sommaValoriFinestra)
            lista_finestre.append(tupla)
            
    lista_finestre_sorted = sorted(lista_finestre, key=lambda i: i[1], reverse=True)[:NUMERO_SERPENTI]
    print(f"sorted = {lista_finestre_sorted}")
    
    percorsi = []
    
    value = -1
    i = 0
    max = 0

    while value <= 0:
        idx = 0
        for serpente in range(0, NUMERO_SERPENTI, 2):
            
            percorso_del_serpente = genera_percorso(serpente, idx)
            percorsi.append(percorso_del_serpente)
            idx = idx + 1
            
        print(f"percorsi = {percorsi}")
        value = compute_fitness(percorsi)
        print(value)

    print_data_output(percorsi,"output.txt")